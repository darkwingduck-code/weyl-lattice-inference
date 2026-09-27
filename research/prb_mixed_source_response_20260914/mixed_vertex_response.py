"""Three-channel response for mixed Gamma5/Gamma3 probes.

Charge and Gamma5 remain the interacting channels. Gamma3 is an external
source/readout direction only; no Gamma3 interaction is introduced.
"""
from __future__ import annotations
from dataclasses import dataclass
import ctypes
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LEGACY = REPO / "research" / "prb_lattice_response"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
from lattice_response import AXIAL, GAMMA, MASS, rpa_response
from parameter_sector_sampling import stable_fermi_difference

@dataclass
class ResponseAccumulator:
    bare: np.ndarray
    moment0: np.ndarray
    current_row: np.ndarray
    current_column: np.ndarray
    density: float
    raw_pairs: int
    nonzero_occupation_differences: int
    restored_occupation_differences: int
    charge_vertex_error: float

def local_operators(denominator: int) -> tuple[np.ndarray, ...]:
    """Return (charge, Gamma5, Gamma3) in a magnetic-cell basis."""
    eye = np.eye(denominator)
    return (np.eye(4 * denominator), np.kron(eye, AXIAL),
            np.kron(eye, GAMMA[2]))

def empty_accumulator(nfreq: int, noperators: int = 3) -> ResponseAccumulator:
    return ResponseAccumulator(
        bare=np.zeros((nfreq, noperators, noperators), complex),
        moment0=np.zeros((noperators, noperators), complex),
        current_row=np.zeros((nfreq, noperators), complex),
        current_column=np.zeros((nfreq, noperators), complex),
        density=0.0, raw_pairs=0, nonzero_occupation_differences=0,
        restored_occupation_differences=0, charge_vertex_error=0.0)

def add_accumulators(target: ResponseAccumulator,
                     source: ResponseAccumulator) -> None:
    for name in ("bare", "moment0", "current_row", "current_column"):
        getattr(target, name)[:] += getattr(source, name)
    target.density += source.density
    for name in ("raw_pairs", "nonzero_occupation_differences",
                 "restored_occupation_differences"):
        setattr(target, name, getattr(target, name) + getattr(source, name))
    target.charge_vertex_error = max(target.charge_vertex_error,
                                     source.charge_vertex_error)
def accumulate_spectrum(spec, qshift: int, mu: float, temperature: float,
                        omega, eta: float, *, operators=None,
                        frequency_batch: int = 4) -> ResponseAccumulator:
    """Accumulate one magnetic-Bloch chunk while streaming its kz planes."""
    p = spec.parameters
    if isinstance(qshift, (bool, np.bool_)) or int(qshift) != qshift:
        raise ValueError("qshift must be an integer")
    qshift = int(qshift)
    if qshift == 0 or abs(qshift) >= p.Nkz:
        raise ValueError("Use nonzero |qshift| < Nkz")
    if temperature <= 0 or eta <= 0 or frequency_batch < 1:
        raise ValueError("Positive temperature, eta, and frequency batch required")
    omega = np.asarray(omega, float)
    if omega.ndim != 1 or not np.all(np.isfinite(omega)):
        raise ValueError("Finite one-dimensional frequency grid required")
    denominator = spec.denominator
    ops = tuple(local_operators(denominator) if operators is None else operators)
    dimension = 4 * denominator
    if not ops or any(np.asarray(op).shape != (dimension, dimension) for op in ops):
        raise ValueError("Every local operator must match the magnetic block")
    if np.max(np.abs(ops[0] - np.eye(dimension))) > 1e-14:
        raise ValueError("First operator must be charge")
    q = 2 * np.pi * qshift / p.Nkz
    Q = 2 * np.sin(q / 2)
    out = empty_accumulator(len(omega), len(ops))
    mass = np.kron(np.eye(denominator), MASS)
    gamma3 = np.kron(np.eye(denominator), GAMMA[2])
    energies = spec.energies
    xall = (energies - mu) / temperature
    occupations = np.where(
        xall >= 0, np.exp(-np.minimum(xall, 745)) /
        (1 + np.exp(-np.minimum(xall, 745))),
        1 / (1 + np.exp(np.maximum(xall, -745))))
    out.density = float(np.sum(occupations) / p.volume)
    for i, k in enumerate(spec.kz):
        j = (i + qshift) % p.Nkz
        udag = spec.vectors[:, i].conj().swapaxes(-1, -2)
        v = spec.vectors[:, j]
        vertices = np.asarray([(udag @ op) @ v for op in ops])
        delta = energies[:, j, None, :] - energies[:, i, :, None]
        x = (energies[:, i, :, None] - mu) / temperature
        y = (energies[:, j, None, :] - mu) / temperature
        stable = stable_fermi_difference(x, y)
        legacy = occupations[:, i, :, None] - occupations[:, j, None, :]
        out.restored_occupation_differences += int(np.count_nonzero(
            (legacy == 0) & (stable != 0)))
        out.nonzero_occupation_differences += int(np.count_nonzero(stable))
        df = stable / p.volume
        out.raw_pairs += delta.size
        current_operator = (p.wilson * np.sin(k + q / 2) * mass
                            + p.hopping * np.cos(k + q / 2) * gamma3)
        current = (udag @ current_operator) @ v
        out.charge_vertex_error = max(out.charge_vertex_error, float(
            np.max(np.abs(Q * current - delta * vertices[0]))))
        dflat, fflat = delta.ravel(), df.ravel()
        vflat, jflat = vertices.reshape(len(ops), -1), current.ravel()
        weights = np.empty((len(ops), len(ops), dflat.size), complex)
        for a in range(len(ops)):
            for b in range(len(ops)):
                weights[a, b] = fflat * vflat[a] * vflat[b].conj()
                out.moment0[a, b] += np.sum(weights[a, b])
        jr = np.asarray([fflat * jflat * vflat[b].conj()
                         for b in range(len(ops))])
        jc = np.asarray([fflat * vflat[a] * jflat.conj()
                         for a in range(len(ops))])
        for first in range(0, len(omega), frequency_batch):
            last = min(first + frequency_batch, len(omega))
            resolvent = 1 / (omega[first:last, None] + 1j * eta
                             - dflat[None, :])
            for a in range(len(ops)):
                for b in range(len(ops)):
                    out.bare[first:last, a, b] += resolvent @ weights[a, b]
            for a in range(len(ops)):
                out.current_row[first:last, a] += resolvent @ jr[a]
                out.current_column[first:last, a] += resolvent @ jc[a]
    return out
def dress_response(bare, Q: float, g: float, ua: float) -> np.ndarray:
    """Dress arbitrary external operators through charge/Gamma5 only."""
    bare = np.asarray(bare, complex)
    if bare.shape[-1] != bare.shape[-2] or bare.shape[-1] < 2:
        raise ValueError("Square response with charge/Gamma5 first required")
    if Q == 0 or g < 0 or ua < 0:
        raise ValueError("Nonzero Q and nonnegative interactions required")
    interaction = np.diag([g / Q**2, ua])
    flat = bare.reshape((-1,) + bare.shape[-2:])
    result = np.empty_like(flat)
    for i, matrix in enumerate(flat):
        screened = np.linalg.solve(np.eye(2) - interaction @ matrix[:2, :2],
                                   interaction)
        result[i] = matrix + matrix[:, :2] @ screened @ matrix[:2, :]
    return result.reshape(bare.shape)

def contract_mixed(response, lambda_readout: float,
                   lambda_source: float | None = None) -> np.ndarray:
    """Contract (charge,Gamma5,Gamma3) to (charge,mixed) input/output."""
    response = np.asarray(response, complex)
    if response.shape[-2:] != (3, 3):
        raise ValueError("Three-operator response required")
    if lambda_source is None:
        lambda_source = lambda_readout
    lr, ls = float(lambda_readout), float(lambda_source)
    if not np.isfinite(lr) or not np.isfinite(ls):
        raise ValueError("Finite real mixture coefficients required")
    readout = np.array([[1., 0.], [0., 1.], [0., lr]])
    source = np.array([[1., 0.], [0., 1.], [0., ls]])
    return np.einsum("ia,...ij,jb->...ab", readout.conj(), response, source)

def apply_transfer(response, readout_gain, source_gain) -> np.ndarray:
    result = np.asarray(response, complex).copy()
    d, s = np.asarray(readout_gain), np.asarray(source_gain)
    result[..., 1, :] *= d[..., None] if d.ndim else d
    result[..., :, 1] *= s[..., None] if s.ndim else s
    return result

def closure(response) -> np.ndarray:
    response = np.asarray(response, complex)
    if np.any(response[..., 1, 1] == 0):
        raise ValueError("Closure undefined at zero mixed response")
    return response[..., 0, 1] * response[..., 1, 0] / response[..., 1, 1]

def ward_errors(acc: ResponseAccumulator, omega, eta: float, Q: float):
    z = np.asarray(omega, float) + 1j * eta
    left = z[:, None] * acc.bare[:, 0, :] - Q * acc.current_row - acc.moment0[0]
    right = z[:, None] * acc.bare[:, :, 0] - Q * acc.current_column - acc.moment0[:, 0]
    return {"left": float(np.max(np.abs(left))),
            "right": float(np.max(np.abs(right)))}

def weighted_passivity_minimum(response, omega, eta: float) -> float:
    values = []
    for w, matrix in zip(np.asarray(omega), np.asarray(response)):
        zchi = (w + 1j * eta) * matrix
        loss = -(zchi - zchi.conj().T) / (2j)
        values.append(float(np.linalg.eigvalsh(loss).min()))
    return min(values)

def scaled_max_error(actual, reference) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))
                        / (1 + np.abs(reference))))

def available_memory_bytes() -> int | None:
    if sys.platform != "win32":
        return None
    class S(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong)]
    status = S(); status.length = ctypes.sizeof(status)
    return int(status.available) if ctypes.windll.kernel32.GlobalMemoryStatusEx(
        ctypes.byref(status)) else None

def process_rss_bytes() -> int | None:
    if sys.platform != "win32":
        return None
    class C(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                    ("peak_ws", ctypes.c_size_t), ("ws", ctypes.c_size_t),
                    ("qpp", ctypes.c_size_t), ("qp", ctypes.c_size_t),
                    ("qnp", ctypes.c_size_t), ("qn", ctypes.c_size_t),
                    ("page", ctypes.c_size_t), ("peak_page", ctypes.c_size_t)]
    counters = C(); counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = kernel32.GetCurrentProcess()
    function = getattr(kernel32, "K32GetProcessMemoryInfo", None)
    if function is None:
        function = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(C), ctypes.c_ulong]
    function.restype = ctypes.c_int
    ok = function(handle, ctypes.byref(counters), counters.cb)
    return int(counters.ws) if ok else None

def verify_legacy_rpa(bare3, Q, g, ua) -> float:
    dressed = dress_response(bare3, Q, g, ua)
    interaction = np.diag([g / Q**2, ua])
    return max(scaled_max_error(dressed[i, :2, :2],
               rpa_response(bare3[i, :2, :2], interaction))
               for i in range(len(bare3)))
