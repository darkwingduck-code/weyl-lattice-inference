"""Full-band charge/axial RPA on a Peierls-coupled Wilson-Dirac torus.

The axial observable is the local bilinear tau_x, not an exactly conserved
valley population. All four magnetic bands and all particle-hole transitions
are retained. Frequency broadening is a resolvent regulator, not collisions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import expit

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.diag([1., -1.]).astype(complex)
MASS = np.kron(SZ, I2)
AXIAL = np.kron(SX, I2)
SPIN_Z = np.kron(I2, SZ)
GAMMA = tuple(np.kron(SX, s) for s in (SX, SY, SZ))


@dataclass(frozen=True)
class Parameters:
    Lx: int = 4
    Ly: int = 4
    Nkz: int = 24
    mass: float = 0.
    wilson: float = 1.
    hopping: float = 1.
    exchange: float = 1.
    theta_x: float = 0.
    theta_y: float = 0.
    theta_z: float = 0.

    def __post_init__(self):
        if min(self.Lx, self.Ly) < 3 or self.Nkz < 4:
            raise ValueError("Use at least 3x3 transverse sites and 4 kz points")

    @property
    def dim(self):
        return 4 * self.Lx * self.Ly

    @property
    def volume(self):
        return self.Lx * self.Ly * self.Nkz

    def field(self, nphi):
        if int(nphi) != nphi:
            raise ValueError("Magnetic torus flux must be an integer")
        return 2 * np.pi * nphi / (self.Lx * self.Ly)

    def kz_grid(self):
        return (2 * np.pi * np.arange(self.Nkz) + self.theta_z) / self.Nkz - np.pi


def bloch_hamiltonian(k, p=Parameters()):
    k = np.asarray(k, float)
    return (sum(p.hopping * np.sin(k[i]) * GAMMA[i] for i in range(3))
            + (p.mass + p.wilson * np.sum(1 - np.cos(k))) * MASS
            + p.exchange * SPIN_Z)


def transverse_hamiltonian(p, nphi):
    out = np.zeros((p.dim, p.dim), complex)
    field = p.field(nphi)

    def site(x, y):
        i = 4 * (x * p.Ly + y)
        return slice(i, i + 4)

    onsite = (p.mass + 3 * p.wilson) * MASS + p.exchange * SPIN_Z
    links = [-p.wilson * MASS / 2 + .5j * p.hopping * g for g in GAMMA[:2]]
    for x in range(p.Lx):
        for y in range(p.Ly):
            a = site(x, y)
            out[a, a] += onsite
            boundary = np.exp(-1j * field * p.Lx * y) if x == p.Lx - 1 else 1.
            for b, link, phase in (
                (site((x + 1) % p.Lx, y), links[0],
                 np.exp(1j * p.theta_x / p.Lx) * boundary),
                (site(x, (y + 1) % p.Ly), links[1],
                 np.exp(1j * p.theta_y / p.Ly + 1j * field * x)),
            ):
                term = phase * link
                out[b, a] += term
                out[a, b] += term.conj().T
    return out


def hamiltonian(p, xy, kz):
    return xy + np.kron(np.eye(p.Lx * p.Ly),
                        -p.wilson * np.cos(kz) * MASS
                        + p.hopping * np.sin(kz) * GAMMA[2])


@dataclass
class Spectrum:
    parameters: Parameters
    nphi: int
    kz: np.ndarray
    energies: np.ndarray
    vectors: np.ndarray
    hamiltonians: np.ndarray


def diagonalize(p, nphi, site_phases=None):
    xy = transverse_hamiltonian(p, nphi)
    hs = np.array([hamiltonian(p, xy, k) for k in p.kz_grid()])
    if site_phases is not None:
        phase = np.asarray(site_phases, float)
        if phase.shape != (p.Lx * p.Ly,):
            raise ValueError("One phase per transverse site required")
        unitary = np.repeat(np.exp(1j * phase), 4)
        hs = unitary[None, :, None] * hs * unitary.conj()[None, None, :]
    values, vectors = zip(*(eigh(h, check_finite=False) for h in hs))
    return Spectrum(p, int(nphi), p.kz_grid(), np.array(values),
                    np.array(vectors), hs)


def occupations(s, mu, temperature):
    if temperature <= 0:
        raise ValueError("Positive temperature required")
    return expit((mu - s.energies) / temperature)


def density(s, mu, temperature):
    return float(occupations(s, mu, temperature).sum() / s.parameters.volume)


def fixed_density_mu(s, target, temperature):
    return float(brentq(lambda mu: density(s, mu, temperature) - target,
                        s.energies.min() - 2, s.energies.max() + 2,
                        xtol=1e-13, rtol=1e-13))


@dataclass
class Transitions:
    delta: np.ndarray
    cc: np.ndarray
    aa: np.ndarray
    ca: np.ndarray
    conduction: np.ndarray
    Q: float
    q: float
    vertex_error: float
    density_value: float

    def density_response(self, z):
        if complex(z).imag <= 0:
            raise ValueError("Evaluate the retarded response in Im(z)>0")
        d = 1 / (complex(z) - self.delta)
        return np.array([[np.dot(self.cc, d), np.dot(self.ca, d)],
                         [np.dot(self.ca.conj(), d), np.dot(self.aa, d)]])

    def full_response(self, z):
        result = np.zeros((3, 3), complex)
        result[:2, :2] = self.density_response(z)
        d = 1 / (complex(z) - self.delta)
        current = self.delta / self.Q
        result[0, 2] = result[2, 0] = np.dot(self.cc * current, d)
        result[1, 2] = np.dot(self.ca.conj() * current, d)
        result[2, 1] = np.dot(self.ca * current, d)
        result[2, 2] = (np.dot(self.cc * current**2, d)
                        + np.dot(self.cc, self.delta) / self.Q**2)
        return result

    def moments(self):
        def matrix(weight):
            return np.array([[np.dot(self.cc, weight), np.dot(self.ca, weight)],
                             [np.dot(self.ca.conj(), weight), np.dot(self.aa, weight)]])
        return matrix(np.ones_like(self.delta)), matrix(self.delta)

    def ward_errors(self, z, interaction=None):
        response = self.full_response(z)
        if interaction is not None:
            response = rpa_response(response, interaction)
        contraction = np.array([z, 0, -self.Q])
        return (float(np.max(np.abs(contraction @ response))),
                float(np.max(np.abs(response @ contraction))))


def transitions(s, mu, temperature, qshift=1):
    p = s.parameters
    if int(qshift) != qshift or qshift == 0 or abs(qshift) >= p.Nkz:
        raise ValueError("Nonzero integer qshift with |qshift|<Nkz required")
    q = 2 * np.pi * qshift / p.Nkz
    Q = 2 * np.sin(q / 2)
    axial = np.kron(np.eye(p.Lx * p.Ly), AXIAL)
    f = occupations(s, mu, temperature)
    columns = {name: [] for name in ("delta", "cc", "aa", "ca", "conduction")}
    error = 0.
    for i in range(p.Nkz):
        j = (i + qshift) % p.Nkz
        u, v = s.vectors[i], s.vectors[j]
        charge = u.conj().T @ v
        a = u.conj().T @ axial @ v
        delta = s.energies[j][None, :] - s.energies[i][:, None]
        current = u.conj().T @ ((s.hamiltonians[j] - s.hamiltonians[i]) / Q) @ v
        error = max(error, float(np.max(np.abs(Q * current - delta * charge))))
        df = (f[i][:, None] - f[j][None, :]) / p.volume
        columns["delta"].append(delta.ravel())
        columns["cc"].append((df * np.abs(charge)**2).ravel())
        columns["aa"].append((df * np.abs(a)**2).ravel())
        columns["ca"].append((df * charge * a.conj()).ravel())
        columns["conduction"].append(((s.energies[i][:, None] > 0)
                                       & (s.energies[j][None, :] > 0)).ravel())
    return Transitions(**{key: np.concatenate(value) for key, value in columns.items()},
                       Q=Q, q=q, vertex_error=error,
                       density_value=density(s, mu, temperature))


def interaction_matrix(Q, coulomb, axial):
    if Q == 0 or coulomb < 0 or axial < 0:
        raise ValueError("Use nonzero Q and nonnegative repulsive interactions")
    return np.diag([coulomb / Q**2, axial])


def rpa_response(bare, interaction):
    bare = np.asarray(bare, complex)
    coupling = np.asarray(interaction, complex)
    if bare.shape not in ((2, 2), (3, 3)) or coupling.shape != (2, 2):
        raise ValueError("Expected charge/axial (and current) kernel")
    if not np.allclose(coupling, coupling.conj().T):
        raise ValueError("Hermitian interaction required")
    screened = np.linalg.solve(np.eye(2) - coupling @ bare[:2, :2], coupling)
    return bare + bare[:, :2] @ screened @ bare[:2, :]


def loss_matrix(response):
    response = np.asarray(response, complex)
    return -(response - response.conj().T) / (2j)


def parameter_dict(p):
    return asdict(p)


def axial_identity(s, mu, temperature, qshift, z, interaction=None):
    """Check the density columns of the axial continuity equation.

    S=[A,H_average]=i*physical_torque is anti-Hermitian. Only the two
    commuting density source columns have zero equal-time commutator here.
    """
    p = s.parameters
    q = 2 * np.pi * qshift / p.Nkz
    Q = 2 * np.sin(q / 2)
    if abs(Q) < 1e-14 or complex(z).imag <= 0:
        raise ValueError("Nonzero q and retarded frequency required")
    a = np.kron(np.eye(p.Lx * p.Ly), AXIAL)
    occ = occupations(s, mu, temperature)
    rows = np.zeros((3, 2), complex)
    polarization = np.zeros((2, 2), complex)
    commutators = np.zeros(2, complex)
    vertex_error = 0.
    for i in range(p.Nkz):
        j = (i + qshift) % p.Nkz
        hi, hj = s.hamiltonians[i], s.hamiltonians[j]
        u, v = s.vectors[i], s.vectors[j]
        r = u.conj().T @ v
        av = u.conj().T @ a @ v
        difference, average = hj - hi, (hj + hi) / 2
        jv = u.conj().T @ ((a @ difference + difference @ a) / (2 * Q)) @ v
        sv = u.conj().T @ (a @ average - average @ a) @ v
        delta = s.energies[j][None, :] - s.energies[i][:, None]
        vertex_error = max(vertex_error, float(np.max(np.abs(delta * av - Q * jv - sv))))
        df = (occ[i][:, None] - occ[j][None, :]) / p.volume
        factor = df / (z - delta)
        for column, source in enumerate((r, av)):
            for row, observable in enumerate((av, jv, sv)):
                rows[row, column] += np.sum(factor * observable * source.conj())
            commutators[column] += np.sum(df * av * source.conj())
            polarization[0, column] += np.sum(factor * r * source.conj())
            polarization[1, column] += np.sum(factor * av * source.conj())
    if interaction is not None:
        inverse = np.eye(2) - np.asarray(interaction) @ polarization
        rows = np.linalg.solve(inverse.T, rows.T).T
        commutators = np.linalg.solve(inverse.T, commutators)
    with_torque = z * rows[0] - Q * rows[1] - rows[2] - commutators
    without_torque = z * rows[0] - Q * rows[1] - commutators
    return {"vertex_error": vertex_error,
            "ward_error_with_torque": float(np.max(np.abs(with_torque))),
            "ward_error_without_torque": float(np.max(np.abs(without_torque))),
            "equal_time_commutator": float(np.max(np.abs(commutators))),
            "charge_column_torque": [float(rows[2, 0].real), float(rows[2, 0].imag)],
            "axial_column_torque": [float(rows[2, 1].real), float(rows[2, 1].imag)]}
