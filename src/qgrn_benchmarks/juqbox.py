"""Juqbox risk-neutral control with native propagation and gradient kernels.

Q-GRN uses the full endpoint-projection and path-leakage Gauss-Newton matrix.
The native objective and gradient include the Störmer-Verlet column-norm
defect, so the curvature factor is separate from an objective residual.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse

from qgrn.compat import GRNOptions, solve_grn

from .runtime import native_command, native_environment, native_root

TASKS = ("juqbox_risk_neutral_swap02",)


def source_hashes():
    from .io import source_hashes as python_source_hashes

    directory = native_root()
    paths = list(directory.glob("juqbox*.jl")) + list(
        (directory / "third_party/Juqbox.jl/src").glob("*.jl")
    )
    return {
        **python_source_hashes(),
        **{
            str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }


def serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return serializable(vars(obj))
    return obj


class JuqboxTask:
    def __init__(self, task_id, *, log_path):
        start = perf_counter()
        if task_id not in TASKS:
            raise ValueError(f"Unknown Juqbox task {task_id}")
        self.task_id = task_id
        self.loaded_source_hashes = source_hashes()
        self.warm_records = {}
        env = native_environment("modern")
        self.log_path = Path(log_path).resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "w", encoding="utf-8")
        command = native_command("modern", "juqbox_bridge.jl", task_id)
        self.process = subprocess.Popen(
            command,
            cwd=native_root(),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Native Juqbox failed at startup; see {log_path}")
        self.metadata = json.loads(line)["metadata"]
        self.free = np.array(self.metadata["free_indices_zero_based"], dtype=int)
        self.ncoeff = self.metadata["ncoeff"]
        self.lower = np.array(self.metadata["lower"])[self.free]
        self.upper = np.array(self.metadata["upper"])[self.free]
        self.bounds = (self.lower, self.upper)
        self.objective_weight = 1.0
        self.A = sparse.csc_matrix((0, self.free.size))
        self.cache = None
        self.oracle_diagnostics = []
        self.startup_seconds = perf_counter() - start

    def reset(self):
        if source_hashes() != self.loaded_source_hashes:
            raise RuntimeError("A source loaded by this persistent session changed on disk")
        self.request("reset", np.zeros(self.ncoeff))
        self.cache = None
        self.oracle_diagnostics = []

    def warm_method(self, method):
        if method not in ("grn", "ipopt"):
            raise ValueError("Unknown method")
        if method in self.warm_records:
            return dict(self.warm_records[method], reused=True)
        start = perf_counter()
        self.request("oracle", np.zeros(self.ncoeff), derivatives=True, curvature=method == "grn")
        if method == "ipopt":
            self.request("ipopt", np.zeros(self.ncoeff), seconds=1.0, maxiter=0)
        record = dict(
            method=method,
            task=self.task_id,
            seconds=perf_counter() - start,
            reused=False,
            point="zero controls independent of all paired starts",
            native_ipopt_iterations=0 if method == "ipopt" else None,
            source_hashes=self.loaded_source_hashes,
        )
        self.warm_records[method] = record
        self.cache = None
        return record

    def request(self, op, x, **kwargs):
        request = dict(op=op, x=np.asarray(x).tolist(), **kwargs)
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Native Juqbox process exited before replying")
        answer = json.loads(line)
        if not answer["ok"]:
            raise RuntimeError(answer["error"])
        return answer["result"]

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=10)
        self.log.close()

    def pack(self, y):
        y = np.asarray(y, dtype=float)
        if y.shape != self.free.shape:
            raise ValueError("Free coefficient shape mismatch")
        x = np.zeros(self.ncoeff)
        x[self.free] = y
        return x

    def initial(self, seed):
        amplitude = self.metadata["initial_uniform_halfwidth"]
        # Draw all original coefficients, then apply the exact endpoint equalities.
        x = np.random.default_rng(seed).uniform(-amplitude, amplitude, self.ncoeff)
        x[np.setdiff1d(np.arange(self.ncoeff), self.free)] = 0
        return x[self.free]

    def restore_initial(self, y):
        y = np.asarray(y, dtype=float)
        margin = 1e-10 * (self.upper - self.lower)
        restored = np.clip(y, self.lower + margin, self.upper - margin)
        return restored, dict(
            method="fixed endpoint elimination and strict box interior",
            norm=float(np.linalg.norm(restored - y)),
        )

    def feasibility(self, y):
        violation = float(max(np.max(self.lower - y), np.max(y - self.upper), 0.0))
        return dict(primal=violation, box=violation, fixed_endpoints=0.0)

    def grn_oracle(self, y, derivatives=True, beta=0.0):
        y = np.asarray(y, dtype=float)
        dl, du = y - self.lower, self.upper - y
        if np.any(dl <= 0) or np.any(du <= 0):
            return dict(valid=False, reason="coefficient_box_exterior")
        if (
            self.cache is None
            or not np.array_equal(y, self.cache[0])
            or (derivatives and "GN" not in self.cache[1])
        ):
            value = self.request(
                "oracle", self.pack(y), derivatives=derivatives, curvature=derivatives
            )
            self.cache = y.copy(), value
        a = self.cache[1]
        n = y.size
        # A square factor is passed through Jr only as a curvature factor. r=0
        # is a dimensional placeholder; the exact native gradient is supplied.
        Jr = np.empty((0, n))
        r = np.empty(0)
        if derivatives:
            H = np.asarray(a["GN"]).reshape((n, n), order="F")
            H = (H + H.T) / 2
            vals, vecs = np.linalg.eigh(H)
            threshold = 1e-10 * max(float(np.max(np.abs(vals))), 1.0)
            if vals[0] < -threshold:
                raise FloatingPointError("Full path GN matrix is not positive semidefinite")
            Jr = np.sqrt(np.maximum(vals, 0.0) / 2)[:, None] * vecs.T
            r = np.zeros(n)
            diagnostic = dict(
                gn_seconds=a["gn_seconds"],
                min_eigenvalue=float(vals[0]),
                native_vs_sensitivity_gradient=float(
                    np.max(np.abs(np.asarray(a["g"])[self.free] - a["sensitivity_gradient"]))
                ),
                native_vs_sensitivity_objective=abs(a["f"] - a["sensitivity_objective"]),
                sv_norm_defects=a["sv_norm_defects"],
            )
            self.oracle_diagnostics.append(diagnostic)
        scale = (self.upper - self.lower) / 2
        barrier = -beta * float(np.sum(np.log(dl / scale) + np.log(du / scale))) if beta else 0.0
        gb = beta * (1 / du - 1 / dl)
        hb = beta * (1 / du**2 + 1 / dl**2)
        return dict(
            valid=True,
            f=a["f"],
            g=np.asarray(a["g"])[self.free] if derivatives else np.zeros(n),
            r=r,
            Jr=Jr,
            barrier=barrier,
            gb=gb,
            H0=sparse.diags(hb, format="csc"),
            gate_error=a["gate_error"],
            ell=a["gate_error"],
            leakage=a["leakage"],
            tikhonov=a["tikhonov"],
            curvature_factor_only=True,
        )


def run_case(
    task_id,
    seed,
    output,
    method,
    seconds=30,
    target=None,
    *,
    session=None,
    input_json=None,
):
    """Run one paired problem, with warm-up and full returns outside solve budget."""
    if target is not None:
        raise ValueError("This exploration explicitly disables objective target stopping")
    if method not in ("grn", "ipopt"):
        raise ValueError("method must be grn or ipopt")
    output = Path(output).resolve()
    if output.exists() or Path(str(output) + ".assets").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = Path(str(output) + ".assets")
    assets.mkdir()
    hashes = source_hashes()
    started_all = perf_counter()
    task = JuqboxTask(task_id, log_path=assets / "native.log") if session is None else session
    if task.task_id != task_id:
        raise ValueError("Persistent session task does not match run task")
    log_offset = task.log.tell()
    try:
        task.reset()
        if input_json is None:
            y0 = task.initial(seed)
            x0 = task.pack(y0)
        else:
            prepared = json.loads(Path(input_json).read_text(encoding="utf-8-sig"))
            if prepared["task"] != task_id or prepared["seed"] != seed:
                raise ValueError("Paired input differs")
            x0 = np.asarray(prepared["coefficients"], dtype=float)
            y0 = x0[task.free]
        setup_seconds = perf_counter() - started_all
        warm_start = perf_counter()
        warm_record = task.warm_method(method)
        warm_seconds = perf_counter() - warm_start
        task.reset()
        initial_start = perf_counter()
        initial = task.request("oracle", x0, derivatives=False, curvature=False)
        initial_traj = task.request("trajectory", x0, path=str(assets / "initial_trajectory.bin"))
        initial_verify = task.request("verify", x0)
        initial_seconds = perf_counter() - initial_start
        task.cache = None
        begin = perf_counter()
        if method == "grn":
            result = solve_grn(
                task,
                y0,
                GRNOptions(
                    estimator="ratio",
                    ratio_update="step_aware",
                    seconds=float(seconds),
                    maxiter=100000,
                    stop_on_target=False,
                ),
                started=begin,
            )
            xf = task.pack(result["x"])
            history = [dict(event, x=task.pack(event["x"])) for event in result["history"]]
            # Sparse raw oracle arrays are not useful serialization artifacts.
            result.pop("final", None)
        else:
            result = task.request("ipopt", x0, seconds=float(seconds), maxiter=100000)
            xf = np.asarray(result["x"])
            history = result["history"]
        solver_client_wall_seconds = perf_counter() - begin
        solver_seconds = solver_client_wall_seconds if method == "grn" else result["elapsed"]
        validation_start = perf_counter()
        final = task.request("oracle", xf, derivatives=False, curvature=False)
        trajectory = task.request("trajectory", xf, path=str(assets / "returned_trajectory.bin"))
        final_verify = task.request("verify", xf)
        return_validation_seconds = perf_counter() - validation_start
        feasibility = float(
            max(
                np.max(np.array(task.metadata["lower"]) - xf),
                np.max(xf - np.array(task.metadata["upper"])),
                0,
            )
        )
        final_hashes = source_hashes()
        raw = dict(
            schema="juqbox_author_grn_exploration_v1",
            task=task_id,
            seed=int(seed),
            method=method,
            budget_seconds=float(seconds),
            target=None,
            stop_on_target=False,
            stopping_policy="user_requested_time_budget_only",
            initial=dict(
                final_objective=initial["f"],
                **initial,
                coefficients=x0,
                trajectory=initial_traj,
                independent=initial_verify,
            ),
            returned=dict(
                final_objective=final["f"],
                **final,
                coefficients=xf,
                trajectory=trajectory,
                feasibility=feasibility,
                independent=final_verify,
            ),
            setup_seconds=setup_seconds,
            warm_seconds=warm_seconds,
            initial_validation_seconds=initial_seconds,
            warm_record=warm_record,
            shared_session=session is not None,
            session_startup_seconds=task.startup_seconds,
            solver_seconds=solver_seconds,
            return_validation_seconds=return_validation_seconds,
            solver_client_wall_seconds=solver_client_wall_seconds,
            history_time_basis="solver",
            initial_elapsed_seconds=0.0,
            returned_elapsed_seconds=solver_seconds + return_validation_seconds,
            complete_seconds=perf_counter() - started_all,
            status=result["status"],
            history=history,
            return_valid=bool(
                np.isfinite(final["f"])
                and feasibility <= 1e-7
                and abs(initial_verify["f"] - initial["f"]) <= 1e-9
                and abs(final_verify["f"] - final["f"]) <= 1e-9
            ),
            initial_coordinate_sha256=hashlib.sha256(
                np.asarray(x0, dtype="<f8").tobytes()
            ).hexdigest(),
            metadata=task.metadata,
            source_hashes=hashes,
            source_hashes_after=final_hashes,
            source_unchanged=hashes == final_hashes,
            result=result,
            grn_oracle_diagnostics=task.oracle_diagnostics,
            budget_semantics="warm oracle before solve; include per-iteration IPC; complete setup and trajectory returns reported separately",
            ipopt_diagnostic_override="author derivative_test disabled after independent full-grid gradient validation",
            grn_model="exact native f/g; full endpoint-projection/path GN; SV norm-defect linearized exactly",
        )
        if session is not None:
            task.log.flush()
            (assets / "native.log").write_bytes(task.log_path.read_bytes()[log_offset:])
        output.write_text(
            json.dumps(serializable(raw), indent=2, allow_nan=False), encoding="utf-8"
        )
        return raw
    except Exception as error:
        failure = dict(
            schema="juqbox_author_grn_exploration_v1",
            task=task_id,
            seed=int(seed),
            method=method,
            budget_seconds=float(seconds),
            status="ERROR",
            return_valid=False,
            error=f"{type(error).__name__}: {error}",
            complete_seconds=perf_counter() - started_all,
            source_hashes=hashes,
            metadata=task.metadata,
            history=serializable(locals().get("history", [])),
        )
        output.write_text(json.dumps(failure, indent=2, allow_nan=False), encoding="utf-8")
        raise
    finally:
        if session is None:
            task.close()
