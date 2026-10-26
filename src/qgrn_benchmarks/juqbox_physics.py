"""Independent original-model coefficients and physical parameter checks."""

import numpy as np

TASKS = ("juqbox_risk_neutral_swap02", "juqbox_guarded_cnot")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message, atol=1e-10, rtol=1e-10):
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    require(
        aa.shape == bb.shape and np.all(np.isfinite(aa)) and np.all(np.isfinite(bb)),
        message + ": shape/finite mismatch",
    )
    require(np.allclose(aa, bb, atol=atol, rtol=rtol), message + ": value mismatch")
    return float(np.max(np.abs(aa - bb), initial=0.0))


def controls(meta, x, times):
    """Independent vectorized quadratic carrier B-splines, author coefficient order."""
    times = np.asarray(times)
    D = int(meta["D1"])
    T = float(meta["T_ns"])
    nc = len(meta["ne"])
    cf = np.asarray(meta["carrier_angular"]).reshape((nc, 2), order="F")
    dt = T / (D - 2)
    width = 3 * dt
    k = np.clip(np.ceil(times / dt + 2).astype(int), 3, D)
    values = np.zeros((times.size, 2 * nc))
    for a in range(nc):
        for freq in range(2):
            pair = []
            for q in range(2):
                offset = 2 * a * 2 * D + freq * 2 * D + q * D
                f = np.zeros(times.size)
                for shift in range(3):
                    j = k - shift
                    tau = (times - dt * (j - 1.5)) / width
                    basis = (
                        1.125 + 4.5 * tau + 4.5 * tau * tau
                        if shift == 0
                        else 0.75 - 9 * tau * tau
                        if shift == 1
                        else 1.125 - 4.5 * tau + 4.5 * tau * tau
                    )
                    f += np.asarray(x)[offset + j - 1] * basis
                pair.append(f)
            phase = cf[a, freq] * times
            values[:, 2 * a] += pair[0] * np.cos(phase) - pair[1] * np.sin(phase)
            values[:, 2 * a + 1] += pair[0] * np.sin(phase) + pair[1] * np.cos(phase)
    return values


def verify_model(meta):
    task = meta["task"]
    D = int(meta["D1"])
    if task == TASKS[0]:
        n, _E, T, nc, amp, tik = 4, 3, 300.0, 48, 2 * np.pi * 0.012 / 2, 0.01
        a = np.diag(np.sqrt(np.arange(1.0, 4.0)), 1)
        num = np.diag(np.arange(4.0))
        H = -np.pi * 0.2198 * (num @ num - num)
        hs = [a + a.T]
        ha = [a - a.T]
        U0 = np.eye(4)[:, :3]
        V = U0[:, [2, 1, 0]]
        W = np.array([0.0, 0.0, 0.0, 1.0])
        cf = np.array([[0.0, -2 * np.pi * 0.2198]])
        nodes, weights = np.polynomial.legendre.leggauss(9)
        nodes *= 2 * np.pi * 0.01
        weights /= 2
        require(D == 12, "RN requires 12 basis coefficients per envelope")
        expected_halfwidth = amp * 0.05
    elif task == TASKS[1]:
        n, _E, T, nc, amp, tik = 16, 4, 250.0, 200, 0.04, 0.0
        aa = np.diag(np.sqrt(np.arange(1.0, 4.0)), 1)
        a = np.kron(np.eye(4), aa)
        b = np.kron(aa, np.eye(4))
        n1 = a.T @ a
        n2 = b.T @ b
        delta = (5.17839 - 5.06323) / 2
        H = (
            2
            * np.pi
            * (
                delta * n1
                - delta * n2
                - 0.3411 / 2 * (n1 @ n1 - n1)
                - 0.3413 / 2 * (n2 @ n2 - n2)
                + 0.001995 * (a @ b.T + a.T @ b)
            )
        )
        hs = [a + a.T, b + b.T]
        ha = [a - a.T, b - b.T]
        U0 = np.eye(16)[:, [0, 1, 4, 5]]
        V = U0[:, [0, 1, 3, 2]]
        W = np.zeros(16)
        for j in range(4):
            for i in range(4):
                if i >= 2 or j >= 2:
                    W[i + 4 * j] = 1 / 7000 if (i, j) in ((0, 2), (1, 2), (2, 0), (2, 1)) else 1 / 7
        dressed = 2 * np.pi * np.sqrt(delta**2 + 0.001995**2)
        cf = np.array([[dressed, -dressed], [dressed, -dressed]])
        nodes, weights = np.array([0.0]), np.array([1.0])
        require(D == 25, "CNOT requires 25 basis coefficients per envelope")
        expected_halfwidth = amp / 100
    else:
        raise ValueError("Unknown paper model")
    for name, expected in (
        ("H0", H),
        ("U0", U0),
        ("target_real", V),
        ("target_imag", np.zeros_like(V)),
        ("carrier_angular", cf),
    ):
        close(
            np.asarray(meta[name]).reshape(expected.shape, order="F"),
            expected,
            "Paper " + name,
            atol=2e-13,
        )
    for name, expected in (("Hsym", hs), ("Hanti", ha)):
        close(
            [np.asarray(a).reshape((n, n), order="F") for a in meta[name]],
            expected,
            "Paper " + name,
            atol=1e-14,
        )
    close(meta["W"], W, "Paper guarded state weights", atol=1e-15)
    close(meta["nodes_angular"], nodes, "Paper quadrature nodes", atol=1e-14)
    close(
        meta["weights_normalized"],
        weights,
        "Paper quadrature normalization",
        atol=1e-14,
    )
    require(
        meta["T_ns"] == T and meta["ncoeff"] == nc and meta["tik0"] == tik,
        "Paper scalar parameters",
    )
    free = np.asarray([j for j in range(nc) if j % D not in (0, 1, D - 2, D - 1)])
    close(
        meta["free_indices_zero_based"],
        free,
        "Fixed coefficient elimination",
        atol=0.0,
        rtol=0.0,
    )
    expected = np.zeros(nc)
    expected[free] = amp
    close(meta["upper"], expected, "Upper coefficient bounds", atol=0.0, rtol=0.0)
    close(meta["lower"], -expected, "Lower coefficient bounds", atol=0.0, rtol=0.0)
    close(
        meta["initial_uniform_halfwidth"],
        expected_halfwidth,
        "Original initialization scale",
        atol=0.0,
        rtol=0.0,
    )
