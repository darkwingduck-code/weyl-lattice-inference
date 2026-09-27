"""Finite-system meromorphic RPA poles and charge zeros.

This module continues a FINITE Lehmann sum off its real transition poles.
It does not choose a continuum Landau sheet or infer damping from finite eta.
No nonzero transition weight is discarded.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular
from scipy.optimize import brentq


def checked_interaction(interaction):
    u = np.asarray(interaction, complex)
    if u.shape != (2, 2) or not np.isfinite(u).all():
        raise ValueError("Finite 2x2 interaction required")
    if np.max(np.abs(u-u.conj().T)) > 1e-13:
        raise ValueError("Hermitian interaction required")
    if np.linalg.eigvalsh(u).min() < 0:
        raise ValueError("Repulsive positive-semidefinite interaction required")
    return u


class RationalResponse:
    def __init__(self, tr):
        delta = np.asarray(tr.delta, float)
        cc, aa, ca = (np.asarray(getattr(tr, key)) for key in ("cc", "aa", "ca"))
        if delta.ndim != 1 or any(x.shape != delta.shape for x in (cc, aa, ca)):
            raise ValueError("One-dimensional transition arrays of equal size required")
        if not all(np.isfinite(x).all() for x in (delta, cc, aa, ca)):
            raise ValueError("Finite transition data required")
        if np.iscomplexobj(cc) and np.any(cc.imag != 0) or np.iscomplexobj(aa) and np.any(aa.imag != 0):
            raise ValueError("Real diagonal weights required")
        cc, aa = cc.real, aa.real
        active = (cc != 0) | (aa != 0) | (ca != 0)
        self.original_count = len(delta)
        self.zero_weight_count = int((~active).sum())
        self.delta = delta[active]
        self.cc, self.aa, self.ca = cc[active], aa[active], ca[active].astype(complex)
        self.sign = np.sign(self.delta)
        if np.any(self.sign == 0):
            raise ValueError("Nonzero weight at exactly zero transition energy")
        if np.any(self.sign*self.cc < 0) or np.any(self.sign*self.aa < 0):
            raise ValueError("Weights violate monotone equilibrium occupation signs")
        a, b, c = self.sign*self.cc, self.sign*self.aa, self.sign*self.ca
        scale = np.maximum(a, b)
        if np.any(scale <= 0):
            raise ValueError("Mixed weight without a diagonal weight")
        # Relative rank-one check, with scaling before products to avoid underflow.
        defect = np.abs((a/scale)*(b/scale)-np.abs(c/scale)**2)
        if np.max(defect, initial=0.) > 2e-12:
            raise ValueError("Each raw transition must have a rank-one weight")
        g = np.empty((2, len(a)), complex)
        choose_c = a >= b
        g[0, choose_c] = np.sqrt(a[choose_c])
        g[1, choose_c] = c[choose_c].conj()/g[0, choose_c]
        g[1, ~choose_c] = np.sqrt(b[~choose_c])
        g[0, ~choose_c] = c[~choose_c]/g[1, ~choose_c]
        self.G = g
        self.factorization_max_error = float(max(
            np.max(np.abs(self.sign*np.abs(g[0])**2-self.cc), initial=0.),
            np.max(np.abs(self.sign*np.abs(g[1])**2-self.aa), initial=0.),
            np.max(np.abs(self.sign*g[0]*g[1].conj()-self.ca), initial=0.)))
        self.moment0 = np.array([[self.cc.sum(), self.ca.sum()],
                                [self.ca.conj().sum(), self.aa.sum()]])

    def polarization(self, z, derivative=0):
        z = complex(z)
        if not np.isfinite(z) or derivative not in (0, 1):
            raise ValueError("Finite z and derivative 0 or 1 required")
        distance = z-self.delta
        if np.any(distance == 0):
            raise ValueError("Evaluation at an active bare transition pole")
        factor = 1/distance if derivative == 0 else -1/distance**2
        return np.array([[self.cc @ factor, self.ca @ factor],
                         [self.ca.conj() @ factor, self.aa @ factor]])

    def dressed(self, z, interaction):
        u = checked_interaction(interaction)
        pi = self.polarization(z)
        return np.linalg.solve(np.eye(2)-pi@u, pi)

    def cofactor(self, z, interaction):
        u = checked_interaction(interaction)
        if u[0, 1] != 0 or u[1, 0] != 0:
            raise ValueError("The named charge cofactor uses diagonal interactions")
        p, dp = self.polarization(z), self.polarization(z, 1)
        det = p[0, 0]*p[1, 1]-p[0, 1]*p[1, 0]
        ddet = (dp[0, 0]*p[1, 1]+p[0, 0]*dp[1, 1]
                -dp[0, 1]*p[1, 0]-p[0, 1]*dp[1, 0])
        vc, ua = u[0, 0].real, u[1, 1].real
        return dict(determinant=1-vc*p[0, 0]-ua*p[1, 1]+vc*ua*det,
                    numerator=p[0, 0]-ua*det,
                    determinant_derivative=-vc*dp[0, 0]-ua*dp[1, 1]+vc*ua*ddet,
                    numerator_derivative=dp[0, 0]-ua*ddet)


def transition_modes(tr, interaction, max_transitions=2500):
    """Hermitian realization, with ALL nonzero finite transition weights.

    H=abs(epsilon)+G^dagger U G=C^dagger C; A=C sign(epsilon) C^dagger.
    F=G C^{-1} gives chi=F (z-A)^{-1} A F^dagger.
    Ill-conditioned Cholesky errors are reported, never repaired by a cutoff.
    """
    response = RationalResponse(tr)
    u = checked_interaction(interaction)
    n = len(response.delta)
    if n == 0 or n > max_transitions:
        raise ValueError("Dense realization size outside the explicit memory bound")
    h = np.diag(np.abs(response.delta)).astype(complex)+response.G.conj().T@u@response.G
    c = cholesky(h, lower=False, check_finite=False)
    hermitian = (c*response.sign[None, :])@c.conj().T
    hermitian_defect = float(np.max(np.abs(hermitian-hermitian.conj().T)))
    values, vectors = eigh(hermitian, check_finite=False)
    f = solve_triangular(c.T, response.G.T, lower=True, check_finite=False).T@vectors
    residues = np.einsum("an,bn,n->nab", f, f.conj(), values)
    weighted_charge = values**2*np.abs(f[0])**2
    probe = .43+.19j
    reconstructed = np.einsum("nab,n->ab", residues, 1/(probe-values))
    error = float(np.max(np.abs(reconstructed-response.dressed(probe, u))))
    return dict(poles=values, residues=residues,
                weighted_charge_weights=weighted_charge,
                amplitudes=f, moment0_error=float(np.max(np.abs(residues.sum(axis=0)))),
                input_moment0_error=float(np.max(np.abs(response.moment0))),
                reconstruction_error=error,
                hermitian_error=hermitian_defect,
                factorization_error=response.factorization_max_error,
                retained_transitions=n, exactly_zero_weights_removed=response.zero_weight_count)


def regular_real_pole(response, interaction, bracket):
    """Certify a sign-changing regular root in an interval free of bare poles.

    This is a finite meromorphic root, not a claim of thermodynamic collectivity.
    """
    left, right = map(float, bracket)
    if not left < right or np.any((response.delta >= left) & (response.delta <= right)):
        raise ValueError("Require ordered bracket containing no active bare pole")
    root = brentq(lambda x:response.cofactor(x, interaction)["determinant"].real,
                  left, right, xtol=1e-13, rtol=1e-13)
    c = response.cofactor(root, interaction)
    derivative = c["determinant_derivative"]
    if abs(derivative) <= 1e-13:
        raise ValueError("Multiple/ill-conditioned root; simple-residue formula not certified")
    residue = c["numerator"]/derivative
    return dict(pole=float(root), charge_residue=complex(residue),
                determinant_residual=float(abs(c["determinant"])),
                charge_numerator=complex(c["numerator"]),
                determinant_derivative=complex(derivative),
                nearest_bare_distance=float(np.min(np.abs(root-response.delta))),
                classification="regular finite-system determinant root; visibility needs residue")


def exterior_real_poles(response, interaction, samples=64):
    """Search positive exterior roots with explicit unresolved boundary/cluster scope."""
    u = checked_interaction(interaction)
    if not len(response.delta):
        return dict(roots=[], status="zero_response", near_edge_unresolved=False)
    moment_budget=1e-10*(1+float(np.sum(abs(response.cc))+np.sum(abs(response.aa))))
    if np.max(abs(response.moment0))>moment_budget:
        raise ValueError("Commuting-density M0=0 required for monotone exterior search")
    edge = float(np.max(response.delta))
    radius = float(np.max(np.abs(response.delta)))
    ev, rot = np.linalg.eigh(u)
    sq = (rot*np.sqrt(ev))@rot.conj().T
    shift = float(np.linalg.eigvalsh(sq@response.G@response.G.conj().T@sq).max())
    margin = max(1e-10, abs(edge)*1e-10)
    report = dict(roots=[], positive_bare_edge=edge, eigenvalue_bound=radius+shift,
                  search_margin=margin, near_edge_unresolved=bool(shift>0),
                  excluded_edge_interval=[edge,edge+margin], samples=samples,
                  scope="Positive exterior only; no conclusion about internal or continuum poles")
    if shift == 0:
        report["status"] = "zero_interaction_no_exterior_determinant_roots"
        return report
    upper = radius+shift+max(1.,radius)*1e-8
    if upper <= edge+margin:
        report["status"] = "unresolved_shift_below_search_margin"
        return report
    nodes = edge+np.geomspace(margin,upper-edge,samples)
    def eigs(x):
        matrix = np.eye(2)-sq@response.polarization(x)@sq
        return np.linalg.eigvalsh(matrix)
    vals = np.array([eigs(x) for x in nodes])
    for channel in range(2):
        for i in range(len(nodes)-1):
            at_left=vals[i,channel]==0
            at_last_right=i==len(nodes)-2 and vals[i+1,channel]==0
            if at_left or at_last_right or vals[i,channel]*vals[i+1,channel] < 0:
                x=(nodes[i] if at_left else nodes[i+1] if at_last_right else
                   brentq(lambda z:eigs(z)[channel],nodes[i],nodes[i+1],xtol=1e-13,rtol=1e-13))
                co=response.cofactor(x,u)
                derivative=co["determinant_derivative"]
                item=dict(pole=float(x),dielectric_channel=channel,
                    determinant_residual=float(abs(co["determinant"])),
                    charge_numerator=complex(co["numerator"]),
                    nearest_bare_distance=float(np.min(np.abs(x-response.delta))))
                item["cofactor_charge_residue"]=(complex(co["numerator"]/derivative)
                    if abs(derivative)>1e-13 else None)
                try:
                    matrix=regular_pole_residue(response,u,x)
                    item["residue_matrix"]=matrix["residue_matrix"]
                    item["residue_status"]="isolated_nullspace_checked"
                    item["nullspace_eigenvalue_residual"]=matrix["eigenvalue_residual"]
                except ValueError as exc:
                    item["residue_status"]="unresolved"
                    item["residue_failure"]=str(exc)
                report["roots"].append(item)
    report["roots"].sort(key=lambda r:r["pole"])
    for i,item in enumerate(report["roots"]):
        item["nearby_root_cluster"]=any(i!=j and abs(item["pole"]-other["pole"])<=1e-8
                                    for j,other in enumerate(report["roots"]))
    report["status"]="bounded_exterior_search_completed"
    return report


def regular_pole_residue(response, interaction, pole, null_tolerance=1e-10,
                         expected_nullity=1, separation_tolerance=1e-8):
    """Matrix residue at a real regular root, including degenerate RPA modes.

    The root must be separated from active bare poles. A small K eigenvalue
    identifies its nullspace; no charge-residue threshold deletes weak modes.
    """
    u = checked_interaction(interaction)
    pole = float(pole)
    if np.any(np.abs(pole-response.delta) <= 1e-11*max(1., abs(pole))):
        raise ValueError("Bare transition collision requires a separate Laurent analysis")
    ev, rot = np.linalg.eigh(u)
    sq = (rot*np.sqrt(ev))@rot.conj().T
    pi, dpi = response.polarization(pole), response.polarization(pole, 1)
    k, dk = np.eye(2)-sq@pi@sq, -sq@dpi@sq
    vals, vectors = np.linalg.eigh(k)
    if expected_nullity not in (1, 2):
        raise ValueError("Expected nullity must be one or two")
    order = np.argsort(np.abs(vals))
    chosen = order[:expected_nullity]
    if np.max(np.abs(vals[chosen])) > null_tolerance:
        raise ValueError("No declared RPA nullspace at the supplied pole")
    if expected_nullity == 1 and abs(vals[order[1]]) <= separation_tolerance:
        raise ValueError("Unresolved nearby dielectric roots; certify multiplicity explicitly")
    selected = np.zeros(2, dtype=bool)
    selected[chosen] = True
    v = vectors[:, selected]
    gram = v.conj().T@dk@v
    minimum = float(np.linalg.eigvalsh(gram).min())
    if pole > 0 and minimum <= 0:
        raise ValueError("Positive-frequency simple-pole metric must be positive")
    if pole < 0 and np.linalg.eigvalsh(gram).max() >= 0:
        raise ValueError("Negative-frequency simple-pole metric must be negative")
    amplitude = pi@sq@v
    residue = amplitude@np.linalg.solve(gram, amplitude.conj().T)
    return dict(residue_matrix=residue, nullity=int(selected.sum()),
                eigenvalue_residual=float(np.max(np.abs(vals[selected]))),
                derivative_minimum=minimum,
                nearest_bare_distance=float(np.min(np.abs(pole-response.delta))),
                charge_coupling_norm=float(np.linalg.norm(amplitude[0])))


def charge_zero_intervals(modes, interval_margin=1e-10):
    """Resolve real zeros of z*chi_cc between visible finite spectral poles.

    Every computed nonzero charge weight remains. Intervals too narrow or
    roots too close to poles for the declared margin are explicitly unresolved.
    Exact zero weights alone do not form part of the scalar visible support.
    The zero near z=0 is marked as introduced by multiplication, not chi=0.
    """
    moment_budget = 1e-10*(1+float(np.sum(np.abs(modes["residues"]))))
    if modes["input_moment0_error"] > moment_budget:
        raise ValueError("Commuting-density M0=0 is required for the z*chi zero identity")
    poles = np.asarray(modes["poles"], float)
    weights = np.asarray(modes["weighted_charge_weights"], float)
    active = weights > 0
    poles, weights = poles[active], weights[active]
    distinct, index = np.unique(poles, return_inverse=True)
    aggregate = np.bincount(index, weights=weights, minlength=len(distinct))
    def transfer(x):
        return float(np.sum(aggregate/(x-distinct)))
    records = []
    for left, right in zip(distinct[:-1], distinct[1:]):
        margin = interval_margin*max(1., abs(left), abs(right))
        record = dict(left_pole=float(left), right_pole=float(right))
        if right-left <= 2*margin:
            record.update(status="unresolved_near_degenerate_interval")
        else:
            a, b = left+margin, right-margin
            if transfer(a)*transfer(b) >= 0:
                record.update(status="unresolved_zero_too_close_to_pole")
            else:
                zero = brentq(transfer, a, b, xtol=1e-13, rtol=1e-13)
                # For commuting density M0=0 and H>0 the central zero is
                # exactly the multiplication zero, since chi_cc(0)<0.
                record.update(zero=float(zero),
                    status="multiplication_zero" if left < 0 < right
                    else "resolved_charge_zero",
                    weighted_transfer_residual=abs(transfer(zero)),
                    nearest_pole_distance=float(min(zero-left,right-zero)))
        records.append(record)
    return records

