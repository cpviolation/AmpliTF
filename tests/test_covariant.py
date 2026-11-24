"""Properties of the covariant spin factors and of the AmpGen-convention lineshapes.

The numerical agreement with AmpGen itself is tested in cpviolation/tfa_examples
(tests/test_ampgen_d02kkpipi.py), against amplitudes computed with AmpGen.
"""

import numpy as np
import pytest

import amplitf.ampgen_lineshapes as agl
import amplitf.covariant as cov
import amplitf.interface as atfi

G = np.diag([-1.0, -1.0, -1.0, 1.0])


def random_momenta(n, masses, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for m in masses:
        p3 = rng.normal(scale=0.5, size=(n, 3))
        out.append(np.column_stack([p3, np.sqrt(m ** 2 + np.sum(p3 ** 2, axis=1))]))
    return out


def boost(p, beta):
    """Boost four-vectors p [N, 4] (x, y, z, t) by velocity beta (3-vector)."""
    beta = np.asarray(beta)
    b2 = beta @ beta
    gamma = 1.0 / np.sqrt(1 - b2)
    bp = p[:, :3] @ beta
    space = p[:, :3] + ((gamma - 1) * bp / b2)[:, None] * beta + (gamma * p[:, 3])[:, None] * beta
    return np.column_stack([space, gamma * (p[:, 3] + bp)])


def c(x):
    return atfi.const(np.asarray(x))


def test_levi_civita_convention():
    eps = cov.levi_civita().numpy()
    assert eps[0, 1, 2, 3] == -1.0  # AmpGen: eps^{xyzt} = +1, all indices lowered
    assert eps[1, 0, 2, 3] == +1.0
    assert eps[0, 0, 2, 3] == 0.0


def test_projectors_and_orbital_tensors():
    pa, pb = random_momenta(100, [0.5, 0.14])
    P, Q = c(pa + pb), c(pa - pb)
    S = cov.spin1_projector(P).numpy()
    np.testing.assert_allclose(np.einsum("kmn,nl,kl->km", S, G, P.numpy()), 0, atol=1e-12)  # transverse
    np.testing.assert_allclose(np.einsum("mn,kmn->k", G, S), 3.0, rtol=1e-12)  # trace
    L1 = cov.orbital_pwave(P, Q).numpy()
    np.testing.assert_allclose(np.einsum("ki,ij,kj->k", L1, G, P.numpy()), 0, atol=1e-12)
    L2 = cov.orbital_dwave(P, Q).numpy()
    np.testing.assert_allclose(L2, np.transpose(L2, (0, 2, 1)), atol=1e-12)  # symmetric
    np.testing.assert_allclose(np.einsum("mn,kmn->k", G, L2), 0, atol=1e-12)  # traceless
    np.testing.assert_allclose(np.einsum("kmn,nl,kl->km", L2, G, P.numpy()), 0, atol=1e-12)


@pytest.mark.parametrize("name", ["S_VV_S", "S_VV_P", "S_VV_D", "S_TV_P", "S_TS_D"])
def test_scalar_vertices_are_lorentz_invariant(name):
    """D -> (a b)(c d): scalar spin factor from the two resonance tensors."""
    pa, pb, pc, pd = random_momenta(50, [0.494, 0.494, 0.14, 0.14])

    def value(ps):
        a, b, cc, d = (c(x) for x in ps)
        r1 = cov.V_SS_P(a + b, a - b, atfi.ones(a[:, 0]), atfi.ones(a[:, 0]))
        if name == "S_TV_P":
            r1 = cov.T_SS_D(a + b, a - b, atfi.ones(a[:, 0]), atfi.ones(a[:, 0]))
        r2 = cov.V_SS_P(cc + d, cc - d, atfi.ones(a[:, 0]), atfi.ones(a[:, 0]))
        if name == "S_TS_D":
            r1 = cov.T_SS_D(a + b, a - b, atfi.ones(a[:, 0]), atfi.ones(a[:, 0]))
            r2 = cov.S_SS_S(cc + d, cc - d, atfi.ones(a[:, 0]), atfi.ones(a[:, 0]))
        return cov.vertex(name, a + b + cc + d, a + b - cc - d, r1, r2).numpy()

    ref = value([pa, pb, pc, pd])
    boosted = value([boost(p, [0.3, -0.2, 0.5]) for p in (pa, pb, pc, pd)])
    np.testing.assert_allclose(boosted, ref, rtol=1e-9, atol=1e-12)
    assert np.all(np.isfinite(ref)) and np.any(np.abs(ref) > 0)


def test_unknown_vertex():
    with pytest.raises(NotImplementedError):
        cov.vertex("S_FF_S", None, None, None, None)


def test_breit_wigner_at_pole():
    m, w, r = 0.896, 0.0503, 1.13
    s1, s2 = c([0.494 ** 2]), c([0.1396 ** 2])
    s = c([m * m])
    for L in (0, 1, 2):
        np.testing.assert_allclose(agl.running_width(s, s1, s2, m, w, r, L).numpy(), w, rtol=1e-12)
        bw = agl.breit_wigner(s, s1, s2, m, w, r, L).numpy()
        ff = agl.form_factor(s, s1, s2, r, L).numpy()
        expected = agl.kfactor(m, w).numpy() * ff / (-1j * m * w)
        np.testing.assert_allclose(bw, expected, rtol=1e-12)


def test_gounaris_sakurai_close_to_breit_wigner_at_pole():
    """At s = m^2 the GS real part of the denominator vanishes, as for the BW."""
    m, w, mpi = 0.77549, 0.1494, 0.13957
    s1 = s2 = c([mpi ** 2])
    s = c([m * m])
    gs = agl.gounaris_sakurai(s, s1, s2, m, w, 1.5, 1, mpi).numpy()
    bw = agl.breit_wigner(s, s1, s2, m, w, 0.0, 1).numpy() * agl.form_factor(s, s1, s2, 1.5, 1).numpy()
    np.testing.assert_allclose(gs, bw, rtol=1e-9)


def test_vector_vertices_are_transverse():
    """V -> V S in S and D wave: the spin tensor is orthogonal to the parent momentum."""
    pa, pb, pc = random_momenta(50, [0.494, 0.14, 0.14])
    a, b, cc = c(pa), c(pb), c(pc)
    one = atfi.ones(a[:, 0])
    v1 = cov.V_SS_P(a + b, a - b, one, one)
    P, Q = a + b + cc, a + b - cc
    for name in ("V_VS_S", "V_VS_D"):
        v = cov.vertex(name, P, Q, v1, one).numpy()
        np.testing.assert_allclose(np.einsum("ki,ij,kj->k", v, G, P.numpy()), 0, atol=1e-12)
        assert np.any(np.abs(v) > 0)


def test_spline_interpolates_knots_and_is_smooth():
    """AmpGen's natural cubic spline: exact at the knots, continuous first derivative,
    zero curvature at the ends, last knot value outside the range."""
    xmin, xmax = 0.2, 2.0
    knots = np.linspace(xmin, xmax, 10)
    values = np.sin(3 * knots) + knots ** 2
    h = knots[1] - knots[0]
    eps = 1e-7
    at = agl.spline(c(knots[1:-1]), values, xmin, xmax).numpy()
    np.testing.assert_allclose(at, values[1:-1], rtol=1e-12)
    lo = agl.spline(c(knots[1:-1] - eps), values, xmin, xmax).numpy()
    hi = agl.spline(c(knots[1:-1] + eps), values, xmin, xmax).numpy()
    np.testing.assert_allclose((hi - at) / eps, (at - lo) / eps, rtol=1e-4, atol=1e-5)
    curv = agl.spline_curvatures(values, xmin, xmax)
    assert curv[0] == curv[-1] == 0.0
    np.testing.assert_allclose(curv[1:-1], (np.roll(values, -1) - 2 * values + np.roll(values, 1))[1:-1] / h ** 2,
                               rtol=0.5)  # close to the finite-difference curvature
    out = agl.spline(c([0.1, 2.5]), values, xmin, xmax).numpy()
    np.testing.assert_allclose(out, values[-1])


def test_gspline_at_pole():
    """At s = m^2 the running width of GSpline is the nominal width."""
    m, w, r = 1.2, 0.4, 1.7
    xmin, xmax = 0.2, 3.0
    values = np.linspace(xmin, xmax, 20) ** 2
    s, s1, s2 = c([m * m]), c([1.02 ** 2]), c([0.1396 ** 2])
    for form, ff in (("EFF", np.exp(-abs(agl.q2(m * m, 1.02 ** 2, 0.1396 ** 2)) * r * r / 2)),
                     ("BlattWeisskopf", agl.form_factor(s, s1, s2, r, 0).numpy())):
        got = agl.gspline(s, s1, s2, m, w, r, 0, values, xmin, xmax, form_factor=form).numpy()
        np.testing.assert_allclose(got, agl.kfactor(m, w).numpy() * ff / (-1j * m * w), rtol=1e-12)


def test_flatte_a0_at_pole():
    m, g, ratio = 0.98, 0.175, 1.2
    mpi, meta, mk = 0.13957, 0.547853, 0.493677
    s = c([1.2])  # above the KK threshold: real phase-space factors
    rho_pieta = np.sqrt((1 - (mpi - meta) ** 2 / 1.2) * (1 - (mpi + meta) ** 2 / 1.2))
    width = g * (rho_pieta + ratio * np.sqrt(1 - 4 * mk ** 2 / 1.2))
    np.testing.assert_allclose(agl.flatte_a0(s, m, g, ratio, mpi, meta, mk).numpy(),
                               1 / (m * m - 1j * m * width - 1.2), rtol=1e-12)
    below = agl.flatte_a0(c([0.9]), m, g, ratio, mpi, meta, mk).numpy()
    width_below = g * (np.sqrt((1 - (mpi - meta) ** 2 / 0.9) * (1 - (mpi + meta) ** 2 / 0.9))
                       + ratio * 1j * np.sqrt(4 * mk ** 2 / 0.9 - 1))
    np.testing.assert_allclose(below, 1 / (m * m - 1j * m * width_below - 0.9), rtol=1e-12)
