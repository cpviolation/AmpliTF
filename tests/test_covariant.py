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


@pytest.mark.parametrize("name", ["S_VV_S", "S_VV_P", "S_VV_D", "S_TV_P"])
def test_scalar_vertices_are_lorentz_invariant(name):
    """D -> (a b)(c d): scalar spin factor from the two resonance tensors."""
    pa, pb, pc, pd = random_momenta(50, [0.494, 0.494, 0.14, 0.14])

    def value(ps):
        a, b, cc, d = (c(x) for x in ps)
        r1 = cov.V_SS_P(a + b, a - b, atfi.ones(a[:, 0]), atfi.ones(a[:, 0]))
        if name == "S_TV_P":
            r1 = cov.T_SS_D(a + b, a - b, atfi.ones(a[:, 0]), atfi.ones(a[:, 0]))
        r2 = cov.V_SS_P(cc + d, cc - d, atfi.ones(a[:, 0]), atfi.ones(a[:, 0]))
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
