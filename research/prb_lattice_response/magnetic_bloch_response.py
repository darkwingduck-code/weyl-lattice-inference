"""Exact rational-flux magnetic Bloch blocks for the frozen Peierls torus.

For B=2*pi*numerator/denominator, the x cell contains denominator sites.
Lx must contain an integer number of these cells. In the convention
psi(x,y)=exp(i*kappa_x*x+i*kappa_y*y)*u[x % denominator], the forward x link
is T_x*exp(-i*kx), while the y dispersion uses ky-B*a on cell site a.
Here kx=(2*pi*ix-theta_x)/Lx and ky=(2*pi*iy-theta_y)/Ly. These signs follow
from the core's H[next,current]=exp(i*A)*(-w*MASS/2+i*t*GAMMA/2).

Only longitudinal probes are supplied. They preserve magnetic Bloch momentum,
so interblock vertices vanish exactly; every band pair within each block is
retained. The denominator is NOT reduced: noncoprime fractions specify an
exact, enlarged magnetic cell. Partial-block spectra support bounded-memory
sums of BARE response; apply RPA only after summing all blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import operator

import numpy as np

try:
    from .lattice_response import (
        AXIAL, GAMMA, MASS, SPIN_Z, Parameters, Transitions,
        density, fixed_density_mu, occupations,
    )
except ImportError:
    from lattice_response import (
        AXIAL, GAMMA, MASS, SPIN_Z, Parameters, Transitions,
        density, fixed_density_mu, occupations,
    )


def _integer(value, name, minimum=None):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, (bool, np.bool_)) or (minimum is not None and result < minimum):
        raise ValueError(f"Invalid {name}")
    return result


def _flux(numerator, denominator):
    return (_integer(numerator, "numerator"),
            _integer(denominator, "denominator", 1))


def _parameters(p, denominator, Nkx, Nky):
    updates = {}
    if Nkx is not None:
        updates["Lx"] = denominator * _integer(Nkx, "Nkx", 1)
    if Nky is not None:
        updates["Ly"] = _integer(Nky, "Nky", 1)
    p = replace(p, **updates) if updates else p
    for name in ("Lx", "Ly", "Nkz"):
        _integer(getattr(p, name), name, 1)
    if p.Lx % denominator:
        raise ValueError("Lx must be a multiple of the magnetic-cell denominator")
    return p


def _phases(cell_phases, denominator):
    phases = np.zeros(denominator) if cell_phases is None else np.asarray(cell_phases, float)
    if phases.shape != (denominator,) or not np.all(np.isfinite(phases)):
        raise ValueError("One finite real gauge phase per magnetic-cell site required")
    return phases.copy()


def _transverse_blocks(p, numerator, denominator, kx, ky, phases):
    dimension = 4 * denominator
    result = np.zeros((len(kx), dimension, dimension), complex)
    field = 2 * np.pi * numerator / denominator
    link = -p.wilson * MASS / 2 + .5j * p.hopping * GAMMA[0]
    forward = np.exp(-1j * kx)[:, None, None] * link
    for a in range(denominator):
        here = slice(4*a, 4*a+4)
        b = (a+1) % denominator
        there = slice(4*b, 4*b+4)
        angle = ky - field*a
        result[:, here, here] += (
            (p.mass + p.wilson*(3-np.cos(angle)))[:, None, None] * MASS
            + (p.hopping*np.sin(angle))[:, None, None] * GAMMA[1]
            + p.exchange*SPIN_Z)
        # += is essential for denominators 1 and 2: distinct links can share
        # the same matrix block, and must not overwrite one another.
        result[:, there, here] += forward
        result[:, here, there] += forward.conj().swapaxes(-1, -2)
    gauge = np.repeat(np.exp(1j*phases), 4)
    return gauge[None, :, None]*result*gauge.conj()[None, None, :]


def magnetic_bloch_hamiltonian(p: Parameters, numerator: int, denominator: int,
                               kx: float, ky: float, kz: float, *,
                               cell_phases=None) -> np.ndarray:
    """Return one 4*denominator block at physical reduced Bloch momenta."""
    numerator, denominator = _flux(numerator, denominator)
    phases = _phases(cell_phases, denominator)
    transverse = _transverse_blocks(p, numerator, denominator,
                                    np.array([kx]), np.array([ky]), phases)[0]
    return transverse + np.kron(np.eye(denominator),
        -p.wilson*np.cos(kz)*MASS + p.hopping*np.sin(kz)*GAMMA[2])


@dataclass(frozen=True)
class MagneticBlochSpectrum:
    parameters: Parameters
    numerator: int
    denominator: int
    energies: np.ndarray
    vectors: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    block_indices: np.ndarray
    cell_phases: np.ndarray

    @property
    def total_blocks(self):
        return (self.parameters.Lx // self.denominator) * self.parameters.Ly

    @property
    def complete(self):
        return (len(self.block_indices) == self.total_blocks
                and np.array_equal(self.block_indices, np.arange(self.total_blocks)))

    @property
    def nphi(self):
        return self.numerator * self.total_blocks

    @property
    def field(self):
        return 2*np.pi*self.numerator/self.denominator


def magnetic_bloch_spectrum(p: Parameters, numerator: int, denominator: int, *,
                            Nkx=None, Nky=None, cell_phases=None,
                            block_range=None) -> MagneticBlochSpectrum:
    """Exact finite-torus spectrum, optionally restricted to magnetic blocks.

    Nkx and Nky, when supplied, set the effective Lx=denominator*Nkx and Ly=Nky;
    the returned parameters always describe that effective torus. Defaults use
    the supplied Parameters unchanged. The core requires Lx,Ly>=3 and Nkz>=4.

    E has shape (nblocks,Nkz,4*denominator); eigenvector columns have shape
    (nblocks,Nkz,4*denominator,4*denominator). kx/ky are flattened coordinates,
    ky varying fastest. block_range=(start,stop) selects a contiguous subset
    without changing normalization or the full torus. Large calculations should
    use iter_magnetic_bloch_spectra to avoid retaining every eigensystem at once.
    """
    numerator, denominator = _flux(numerator, denominator)
    p = _parameters(p, denominator, Nkx, Nky)
    total = (p.Lx // denominator)*p.Ly
    if block_range is None:
        first, last = 0, total
    else:
        if len(block_range) != 2:
            raise ValueError("block_range must be (start,stop)")
        first = _integer(block_range[0], "block start", 0)
        last = _integer(block_range[1], "block stop", 1)
        if not first < last <= total:
            raise ValueError("block_range must be a nonempty subset of all blocks")
    indices = np.arange(first, last)
    kx = (2*np.pi*(indices // p.Ly)-p.theta_x)/p.Lx
    ky = (2*np.pi*(indices % p.Ly)-p.theta_y)/p.Ly
    kz = p.kz_grid()
    phases = _phases(cell_phases, denominator)
    transverse = _transverse_blocks(p, numerator, denominator, kx, ky, phases)
    mass = np.kron(np.eye(denominator), MASS)
    gamma_z = np.kron(np.eye(denominator), GAMMA[2])
    longitudinal = (-p.wilson*np.cos(kz)[:, None, None]*mass
                    + p.hopping*np.sin(kz)[:, None, None]*gamma_z)
    values, vectors = np.linalg.eigh(transverse[:, None] + longitudinal[None, :])
    return MagneticBlochSpectrum(p, numerator, denominator, values, vectors,
                                 kx, ky, kz, indices, phases)


def iter_magnetic_bloch_spectra(p: Parameters, numerator: int, denominator: int, *,
                                Nkx=None, Nky=None, cell_phases=None,
                                blocks_per_chunk=1):
    """Yield exact block subsets; all subsets share the FULL torus volume.

    Sum their density/moments/bare response, then apply the interaction once.
    A fixed-density solve must likewise use total density across all chunks.
    """
    numerator, denominator = _flux(numerator, denominator)
    p = _parameters(p, denominator, Nkx, Nky)
    blocks_per_chunk = _integer(blocks_per_chunk, "blocks_per_chunk", 1)
    total = (p.Lx // denominator)*p.Ly
    for first in range(0, total, blocks_per_chunk):
        yield magnetic_bloch_spectrum(p, numerator, denominator,
            cell_phases=cell_phases, block_range=(first,min(first+blocks_per_chunk,total)))


def magnetic_bloch_density(spec: MagneticBlochSpectrum, mu: float, T: float) -> float:
    """Full density, or a block subset's contribution per full lattice cell."""
    return density(spec, mu, T)


def magnetic_bloch_mu(spec: MagneticBlochSpectrum, target: float, T: float) -> float:
    """Fixed-density chemical potential; requires a complete spectrum."""
    if not spec.complete:
        raise ValueError("Fixed-density mu requires all magnetic Bloch blocks")
    return fixed_density_mu(spec, target, T)


def magnetic_bloch_transitions(spec: MagneticBlochSpectrum, mu: float, T: float,
                               qshift: int = 1) -> Transitions:
    """All longitudinal transitions, including zero-weight pairs and all bands.

    Partial spectra contribute additively to the BARE response and moments.
    The independent analytic Peierls current verifies the finite-q vertex.
    Flattening order is (kz, selected magnetic block, initial band, final band).
    """
    p = spec.parameters
    shift = _integer(qshift, "qshift")
    if shift == 0 or abs(shift) >= p.Nkz:
        raise ValueError("Nonzero integer qshift with |qshift|<Nkz required")
    q = 2*np.pi*shift/p.Nkz
    Q = 2*np.sin(q/2)
    dimension = 4*spec.denominator
    pairs_per_plane = len(spec.block_indices)*dimension**2
    count = p.Nkz*pairs_per_plane
    columns = {name: np.empty(count,dtype=dtype) for name,dtype in (
        ("delta",float),("cc",float),("aa",float),("ca",complex),("conduction",bool))}
    eye = np.eye(spec.denominator)
    axial = np.kron(eye,AXIAL)
    mass, gamma_z = np.kron(eye,MASS), np.kron(eye,GAMMA[2])
    f = occupations(spec,mu,T)
    error = 0.
    for i,k in enumerate(spec.kz):
        j = (i+shift) % p.Nkz
        udag = spec.vectors[:,i].conj().swapaxes(-1,-2)
        v = spec.vectors[:,j]
        charge, a = udag @ v, (udag @ axial) @ v
        delta = spec.energies[:,j,None,:]-spec.energies[:,i,:,None]
        current_operator = (p.wilson*np.sin(k+q/2)*mass
                            + p.hopping*np.cos(k+q/2)*gamma_z)
        current = (udag @ current_operator) @ v
        error = max(error,float(np.max(np.abs(Q*current-delta*charge))))
        df = (f[:,i,:,None]-f[:,j,None,:])/p.volume
        target = slice(i*pairs_per_plane,(i+1)*pairs_per_plane)
        columns["delta"][target] = delta.ravel()
        columns["cc"][target] = (df*np.abs(charge)**2).ravel()
        columns["aa"][target] = (df*np.abs(a)**2).ravel()
        columns["ca"][target] = (df*charge*a.conj()).ravel()
        columns["conduction"][target] = (
            (spec.energies[:,i,:,None]>0)&(spec.energies[:,j,None,:]>0)).ravel()
    return Transitions(**columns,Q=Q,q=q,vertex_error=error,
                       density_value=magnetic_bloch_density(spec,mu,T))