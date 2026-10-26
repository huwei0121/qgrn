"""Independent value-only replay of the original risk-neutral SV discretization.

All uncertainty nodes and all real-endpoint/imaginary-midpoint leakage terms
are retained. The original finite Neumann implicit solve is reproduced; no
optimizer, sensitivities, Jacobian, or approximate smaller time grid is used.
"""

from __future__ import annotations

import numpy as np

from .juqbox_physics import controls, verify_model


def replay(meta, coefficients):
    verify_model(meta)
    x = np.asarray(coefficients, float)
    n, e, steps = (
        int(meta["physical_dimension"]),
        int(meta["computational_dimension"]),
        int(meta["nsteps"]),
    )
    h = float(meta["T_ns"]) / steps
    half = h / 2
    nodes, weights = (
        np.asarray(meta["nodes_angular"]),
        np.asarray(meta["weights_normalized"]),
    )
    h0 = np.asarray(meta["H0"]).reshape((n, n), order="F")
    hs = np.asarray([np.asarray(v).reshape((n, n), order="F") for v in meta["Hsym"]])
    ha = np.asarray([np.asarray(v).reshape((n, n), order="F") for v in meta["Hanti"]])
    waveform = controls(meta, x, np.arange(2 * steps + 1) * half)
    kgrid = h0 + np.einsum("ta,aij->tij", waveform[:, ::2], hs)
    sgrid = np.einsum("ta,aij->tij", waveform[:, 1::2], ha)
    perturbation = np.zeros((len(nodes), n, n))
    for j in range(1, n):
        perturbation[:, j, j] = 0.01 * nodes * 10.0 ** (j - 1)
    initial = np.asarray(meta["U0"]).reshape((n, e), order="F")
    target = (np.asarray(meta["target_real"]) + 1j * np.asarray(meta["target_imag"])).reshape(
        (n, e), order="F"
    )
    u = np.repeat(initial[None], len(nodes), axis=0)
    v = np.zeros_like(u)
    leak = np.zeros(len(nodes))
    guarded = np.asarray(meta["W"])[None, :, None]
    terms = int(meta["neumann_terms"])

    def neumann(matrix, rhs):
        term, total = rhs.copy(), rhs.copy()
        for _ in range(terms):
            term = half * (matrix @ term)
            total += term
        return total

    for index in range(steps):
        k0, km, k1 = (kgrid[2 * index + j] + perturbation for j in (0, 1, 2))
        s0, sm, s1 = sgrid[2 * index : 2 * index + 3]
        ell = neumann(sm, km @ u + sm @ v)
        vm = v + half * ell
        first = s0 @ u - k0 @ vm
        second = neumann(s1, s1 @ (u + half * first) - k1 @ vm)
        un = u + half * (first + second)
        vn = v + half * (ell + km @ un + sm @ vm)
        leak += np.sum(guarded * (0.5 * u * u + 0.5 * un * un + vm * vm), axis=(1, 2)) / steps
        u, v = un, vn
    terminal = u - 1j * v
    overlap = np.sum(target.conj()[None] * terminal, axis=(1, 2))
    gate = 1 - np.abs(overlap) ** 2 / e**2
    tik = float(meta["tik0"]) * float(x @ x) / int(meta["tik_denominator"])
    return dict(
        objective=float(weights @ (gate + leak) + tik),
        gate_error=float(weights @ gate),
        leakage=float(weights @ leak),
        tikhonov=tik,
        original_steps=steps,
        uncertainty_nodes=len(nodes),
        neumann_terms=terms,
        objective_provenance="independent_original_SV_after_coefficient_box_projection",
    )
