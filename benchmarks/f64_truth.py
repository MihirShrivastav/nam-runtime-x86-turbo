"""Independent float64 implementation of the A2 forward pass, for judging precision.

The network is feed-forward, so it can be computed layer by layer over the whole signal (no sample
loop). The input is preceded by the 6347 silent samples NAM Core prewarms with.

    python benchmarks/f64_truth.py <model.nam> <slim> <input.raw> <out.raw> [--max-samples N]

Same weight order as NAM Core's a2_fast.cpp (see src/a2_weights.h). Not fast, not a deliverable:
it answers "which implementation is closer to the exact arithmetic?".
"""

import json
import sys

import numpy as np

K = [6] * 14 + [15, 15] + [6] * 7
D = [1, 3, 7, 17, 41, 101, 239] * 2 + [1, 13] + [1, 3, 7, 17, 41, 101, 239]
HEAD_K = 16
PREWARM = 1 + sum((k - 1) * d for k, d in zip(K, D)) + HEAD_K - 1  # 6347


def load(path: str, slim: float):
    j = json.load(open(path))
    if j.get("architecture") == "SlimmableContainer":
        subs = j["config"]["submodels"]
        idx = next((i for i, s in enumerate(subs) if slim < s["max_value"]), len(subs) - 1)
        j = subs[idx]["model"]
    C = j["config"]["layers"][0]["channels"]
    return C, np.asarray(j["weights"], dtype=np.float32).astype(np.float64)


def shift(a: np.ndarray, s: int) -> np.ndarray:
    """a[:, t - s] with zeros before the start."""
    if s == 0:
        return a
    out = np.zeros_like(a)
    out[:, s:] = a[:, : a.shape[1] - s]
    return out


def forward(C: int, w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xin = np.concatenate([np.zeros(PREWARM), x.astype(np.float64)])
    N = xin.size
    pos = 0

    def take(n):
        nonlocal pos
        v = w[pos : pos + n]
        pos += n
        return v

    X = take(C)[:, None] * xin[None, :]
    head_sum = np.zeros((C, N))
    for l in range(23):
        k, d = K[l], D[l]
        conv = take(C * C * k).reshape(C, C, k)  # [out, in, tap]
        cb, mix = take(C), take(C)
        l1 = take(C * C).reshape(C, C)  # [out, in]
        l1b = take(C)
        Z = cb[:, None] + mix[:, None] * xin[None, :]
        for t in range(k):
            Z += conv[:, :, t] @ shift(X, (k - 1 - t) * d)
        A = np.where(Z > 0, Z, 0.01 * Z)
        head_sum += A
        X = X + l1 @ A + l1b[:, None]
    hw = take(C * HEAD_K).reshape(C, HEAD_K)  # [in, tap]
    hb, hs = take(1)[0], take(1)[0]
    assert pos == w.size, "weight stream not consumed exactly"
    y = np.full(N, hb)
    for t in range(HEAD_K):
        y += hw[:, t] @ shift(head_sum, HEAD_K - 1 - t)
    return (hs * y)[PREWARM:]


def main() -> None:
    model, slim, inp, out = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    x = np.fromfile(inp, dtype="<f4")
    if "--max-samples" in sys.argv:
        x = x[: int(sys.argv[sys.argv.index("--max-samples") + 1])]
    C, w = load(model, slim)
    forward(C, w, x).tofile(out)  # float64 output
    print(f"wrote {out}: {x.size} float64 samples, {C} channels")


if __name__ == "__main__":
    main()
