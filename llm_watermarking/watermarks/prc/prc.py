from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LDPCPRC0Params:
    """
    Parameters for LDPC-PRC0 (Construction 2 in the PRC paper).

    Notation follows the paper:
      - n: codeword length (bits)
      - g: generator matrix columns
      - t: row weight for sparse parity checks (each P row has exactly t ones)
      - r: number of parity checks (rows of P)
      - eta: noise rate for Encode (Bernoulli bit flips)
      - zeta: decoding threshold slack (Decode accepts if syndrome weight < (1/2 - zeta) r)
    """

    n: int
    g: int
    t: int
    r: int
    eta: float
    zeta: float

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if not (0 <= self.eta < 0.5):
            raise ValueError("eta must be in [0, 0.5)")
        if not (0 <= self.zeta < 0.5):
            raise ValueError("zeta must be in [0, 0.5)")
        if self.r <= 0 or self.r > self.n:
            raise ValueError("r must be in [1, n]")
        if self.t <= 0 or self.t > self.n:
            raise ValueError("t must be in [1, n]")
        if self.g <= 0:
            raise ValueError("g must be positive")


@dataclass(frozen=True)
class LDPCPRC0SecretKey:
    params: LDPCPRC0Params
    P_rows: Tuple[int, ...]  # r bitmasks of length n (row i has exactly t bits set)
    z: int  # length-n mask


@dataclass(frozen=True)
class LDPCPRC0PublicKey:
    params: LDPCPRC0Params
    G_cols: Tuple[int, ...]  # g length-n vectors (columns) s.t. P * G = 0
    z: int  # length-n mask (same as in secret key)


def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_t_sparse_row(n: int, t: int, gen: np.random.Generator) -> int:
    idx = gen.choice(n, size=t, replace=False)
    row = 0
    for j in idx:
        row |= 1 << int(j)
    return row


def _sample_uniform_vector(n: int, gen: np.random.Generator) -> int:
    bits = gen.integers(0, 2, size=n, dtype=np.uint8)
    v = 0
    for i, b in enumerate(bits):
        if int(b):
            v |= 1 << i
    return v


def _sample_bernoulli_vector(n: int, p: float, gen: np.random.Generator) -> int:
    if p <= 0.0:
        return 0
    if p >= 1.0:
        return (1 << n) - 1
    bits = (gen.random(n) < p).astype(np.uint8)
    v = 0
    for i, b in enumerate(bits):
        if int(b):
            v |= 1 << i
    return v


def _rref_gf2(rows: Sequence[int], n: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Reduced row echelon form over GF(2), represented as Python ints.

    Returns (rref_rows, pivot_cols). rref_rows has length == rank.
    Pivot columns are in 0..n-1, where bit (1<<col) is the pivot.
    """
    mat = list(rows)
    pivot_cols: list[int] = []
    lead_row = 0
    m = len(mat)

    for col in range(n):
        mask = 1 << col
        pivot = None
        for r in range(lead_row, m):
            if mat[r] & mask:
                pivot = r
                break
        if pivot is None:
            continue

        mat[lead_row], mat[pivot] = mat[pivot], mat[lead_row]
        pivot_row = mat[lead_row]

        for r in range(m):
            if r != lead_row and (mat[r] & mask):
                mat[r] ^= pivot_row

        pivot_cols.append(col)
        lead_row += 1
        if lead_row == m:
            break

    return tuple(mat[:lead_row]), tuple(pivot_cols)


def _sample_kernel_vector(
    n: int,
    rref_rows: Sequence[int],
    pivot_cols: Sequence[int],
    gen: np.random.Generator,
) -> int:
    pivot_set = set(pivot_cols)
    v = 0
    for col in range(n):
        if col in pivot_set:
            continue
        if int(gen.integers(0, 2)):
            v |= 1 << col

    for row_int, piv in zip(rref_rows, pivot_cols):
        if (row_int & v).bit_count() & 1:
            v |= 1 << int(piv)

    return v


def _syndrome_weight(P_rows: Sequence[int], y: int) -> int:
    """
    Compute wt(P y) for P given as row bitmasks.
    """
    w = 0
    for row in P_rows:
        w += (row & y).bit_count() & 1
    return w


class LDPCPRC0:
    """
    Zero-bit public-key PRC based on LDPC codes (Construction 2).

    API mirrors the paper:
      - KeyGen -> (sk, pk)
      - Encode(pk) -> x in {0,1}^n (returned as a Python int bitmask)
      - Decode(sk, x) -> 1 or None
    """

    def __init__(self, params: LDPCPRC0Params, seed: Optional[int] = None) -> None:
        self.params = params
        self._gen = _rng(seed)

    def keygen(self) -> Tuple[LDPCPRC0SecretKey, LDPCPRC0PublicKey]:
        p = self.params

        P_rows = tuple(_sample_t_sparse_row(p.n, p.t, self._gen) for _ in range(p.r))
        rref_rows, pivot_cols = _rref_gf2(P_rows, p.n)
        G_cols = tuple(
            _sample_kernel_vector(p.n, rref_rows, pivot_cols, self._gen) for _ in range(p.g)
        )
        z = _sample_uniform_vector(p.n, self._gen)

        sk = LDPCPRC0SecretKey(params=p, P_rows=P_rows, z=z)
        pk = LDPCPRC0PublicKey(params=p, G_cols=G_cols, z=z)
        return sk, pk

    def encode(self, pk: LDPCPRC0PublicKey) -> int:
        p = pk.params
        if p != self.params:
            raise ValueError("public key params mismatch")

        u_bits = self._gen.integers(0, 2, size=p.g, dtype=np.uint8)
        gu = 0
        for bit, col in zip(u_bits, pk.G_cols):
            if int(bit):
                gu ^= int(col)

        e = _sample_bernoulli_vector(p.n, p.eta, self._gen)
        return gu ^ int(pk.z) ^ e

    def decode(self, sk: LDPCPRC0SecretKey, x: int) -> Optional[int]:
        p = sk.params
        if p != self.params:
            raise ValueError("secret key params mismatch")

        y = int(x) ^ int(sk.z)
        w = _syndrome_weight(sk.P_rows, y)
        threshold = (0.5 - float(p.zeta)) * float(p.r)
        return 1 if w < threshold else None

    @staticmethod
    def default_subexp_lpn_params(
        n: int,
        *,
        eta: float = 0.125,
        t_factor: float = 4.0,
        r_fraction: float = 0.99,
    ) -> LDPCPRC0Params:
        """
        Convenience parameters aligned with Section 5.4 / Theorem 2 style choices:
          - r ≈ 0.99 n
          - t = Θ(log n)
          - g = Ω(log^2 n) (we choose ≈ t^2)
          - zeta = r^{-1/4}

        These defaults are meant for research experiments; tune as needed.
        """
        if n <= 1:
            raise ValueError("n must be > 1")

        r = max(1, min(n, int(round(r_fraction * n))))
        t = int(max(1, min(n, round(t_factor * np.log2(n)))))
        g = int(max(1, round(t * t)))
        zeta = min(0.49, float(r) ** (-0.25))
        return LDPCPRC0Params(n=n, g=g, t=t, r=r, eta=eta, zeta=zeta)


def smoke_test(seed: int = 1234) -> None:
    """
    Minimal correctness sanity check (not a cryptographic test).
    """
    # Use toy parameters chosen so Decode succeeds with overwhelming probability.
    params = LDPCPRC0Params(n=256, g=64, t=8, r=128, eta=0.01, zeta=0.01)
    prc = LDPCPRC0(params, seed=seed)
    sk, pk = prc.keygen()

    x = prc.encode(pk)
    assert prc.decode(sk, x) == 1

    # Random string should decode to None with high probability.
    gen = _rng(seed + 1)
    random_x = _sample_uniform_vector(params.n, gen)
    _ = prc.decode(sk, random_x)
