"""Exact zero-field Bloch decomposition of the Peierls Wilson-Dirac torus.

A longitudinal probe preserves transverse Bloch momentum. Its matrix elements
between different transverse blocks vanish by translation symmetry; within
each block all four bands and all 16 particle-hole pairs are retained, without
a spectral-weight cutoff. This module does not approximate nonzero flux.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .lattice_response import (
        AXIAL, GAMMA, MASS, SPIN_Z, Parameters, Transitions,
        density, fixed_density_mu, occupations,
    )
except ImportError:  # Direct scripts and unittest discovery from this directory.
    from lattice_response import (
        AXIAL, GAMMA, MASS, SPIN_Z, Parameters, Transitions,
        density, fixed_density_mu, occupations,
    )


@dataclass(frozen=True)
class BlochSpectrum:
    """Zero-field spectrum, with transverse sites replaced by momentum blocks.

    ``kx`` and ``ky`` have shape (Lx*Ly,), ordered with ky varying fastest.
    ``kz`` has shape (Nkz,). Energies have shape (Lx*Ly, Nkz, 4), and
    eigenvector columns have shape (Lx*Ly, Nkz, 4, 4).
    """

    parameters: Parameters
    energies: np.ndarray
    vectors: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray


def bloch_spectrum(p: Parameters) -> BlochSpectrum:
    """Diagonalize all 4x4 blocks with the torus's exact twist convention."""
    kx_grid = (2 * np.pi * np.arange(p.Lx) - p.theta_x) / p.Lx
    ky_grid = (2 * np.pi * np.arange(p.Ly) - p.theta_y) / p.Ly
    kx, ky = (grid.ravel() for grid in np.meshgrid(
        kx_grid, ky_grid, indexing="ij"))
    kz = p.kz_grid()
    transverse = (
        (p.mass + p.wilson * (3 - np.cos(kx) - np.cos(ky)))[:, None, None]
        * MASS
        + (p.hopping * np.sin(kx))[:, None, None] * GAMMA[0]
        + (p.hopping * np.sin(ky))[:, None, None] * GAMMA[1]
        + p.exchange * SPIN_Z
    )
    longitudinal = (-p.wilson * np.cos(kz)[:, None, None] * MASS
                    + p.hopping * np.sin(kz)[:, None, None] * GAMMA[2])
    hamiltonians = transverse[:, None, :, :] + longitudinal[None, :, :, :]
    energies, vectors = np.linalg.eigh(hamiltonians)
    return BlochSpectrum(p, energies, vectors, kx, ky, kz)


def bloch_density(spec: BlochSpectrum, mu: float, T: float) -> float:
    """Particles per lattice cell, including every band and momentum block."""
    return density(spec, mu, T)


def bloch_mu(spec: BlochSpectrum, target: float, T: float) -> float:
    """Use the same fixed-density solver and tolerance as the Peierls core."""
    return fixed_density_mu(spec, target, T)


def bloch_transitions(spec: BlochSpectrum, mu: float, T: float,
                      qshift: int = 1) -> Transitions:
    """Return the core retarded-response API for a longitudinal lattice probe.

    The independent Peierls midpoint current is used to check the finite-q
    vertex identity. The core Transitions class then supplies the current
    response, contact term, moments, and Ward contractions.

    Arrays are flattened in (kz, transverse block, initial band, final band)
    order. All 16 pairs remain present even when their occupation difference
    or a matrix element vanishes. Scratch arrays cover one kz plane at a time.
    """
    p = spec.parameters
    if int(qshift) != qshift or qshift == 0 or abs(qshift) >= p.Nkz:
        raise ValueError("Nonzero integer qshift with |qshift|<Nkz required")
    qshift = int(qshift)
    q = 2 * np.pi * qshift / p.Nkz
    Q = 2 * np.sin(q / 2)
    f = occupations(spec, mu, T)
    count = p.Nkz * p.Lx * p.Ly * 16
    columns = {name: np.empty(count, dtype=dtype) for name, dtype in (
        ("delta", float), ("cc", float), ("aa", float),
        ("ca", complex), ("conduction", bool))}
    pairs_per_plane = p.Lx * p.Ly * 16
    error = 0.
    for i, k in enumerate(spec.kz):
        j = (i + qshift) % p.Nkz
        udag = spec.vectors[:, i].conj().swapaxes(-1, -2)
        v = spec.vectors[:, j]
        charge = udag @ v
        axial = (udag @ AXIAL) @ v
        delta = spec.energies[:, j, None, :] - spec.energies[:, i, :, None]
        midpoint = k + q / 2
        current_operator = (p.wilson * np.sin(midpoint) * MASS
                            + p.hopping * np.cos(midpoint) * GAMMA[2])
        current = (udag @ current_operator) @ v
        error = max(error, float(np.max(np.abs(Q * current - delta * charge))))
        df = (f[:, i, :, None] - f[:, j, None, :]) / p.volume
        target = slice(i * pairs_per_plane, (i + 1) * pairs_per_plane)
        columns["delta"][target] = delta.ravel()
        columns["cc"][target] = (df * np.abs(charge)**2).ravel()
        columns["aa"][target] = (df * np.abs(axial)**2).ravel()
        columns["ca"][target] = (df * charge * axial.conj()).ravel()
        columns["conduction"][target] = (
            (spec.energies[:, i, :, None] > 0)
            & (spec.energies[:, j, None, :] > 0)).ravel()
    return Transitions(**columns, Q=Q, q=q, vertex_error=error,
                       density_value=bloch_density(spec, mu, T))