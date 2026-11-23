"""Multi-meson (4pi) phase space of the K-matrix: AmpGen and BaBar/Belle approximations."""

import numpy as np
import pytest

import amplitf.interface as atfi
import amplitf.dynamics as atfd

MPI = 0.13957  # as in Resonance.cpp
S = np.linspace(4 * MPI**2, 3.0, 2001)


def above(s):
    return np.sqrt(np.clip(1.0 - 16.0 * MPI * MPI / s, 0, None))


def rho4pi_belle(s):
    """Transcription of the rho3 term of Resonance::amplitude_KMatrix (d02kspipi_toys)."""
    poly = (1.2274 + 0.00370909 / (s * s) - 0.111203 / s - 6.39017 * s
            + 16.8358 * s * s - 21.8845 * s**3 + 11.3153 * s**4)
    return np.where(s <= 1, np.sqrt(1.0 - 16.0 * MPI * MPI) * poly, above(s))


def rho4pi_ampgen(s):
    poly = (0.0005 - 0.0193 * s + 0.1385 * s**2 - 0.2084 * s**3
            - 0.2974 * s**4 + 0.1366 * s**5 + 1.0789 * s**6)
    return np.where(s <= 1, poly, above(s))


def rho(model=None):
    kwargs = {} if model is None else {"model": model}
    return atfd.kmatrix_phsp_multimeson(atfi.const(S), atfi.const(MPI), **kwargs).numpy()


@pytest.mark.parametrize("model, expected", [(None, rho4pi_ampgen), ("ampgen", rho4pi_ampgen),
                                             ("belle", rho4pi_belle)])
def test_values(model, expected):
    r = rho(model)
    np.testing.assert_allclose(r.real, expected(S), rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(r.imag, 0.0)


def test_unknown_model():
    with pytest.raises(ValueError):
        rho("other")


def test_lineshape_passes_model():
    """kmatrix_lineshape uses the requested approximation only below 1 GeV^2."""
    m_poles = atfi.const([0.651, 1.2036, 1.55817, 1.21, 1.82206])
    g_poles = atfi.const([[0.22889, -0.55377, 0, -0.39899, -0.34639],
                          [0.94128, 0.55095, 0, 0.39065, 0.31503],
                          [0.36856, 0.23888, 0.55639, 0.18340, 0.18681],
                          [0.33650, 0.40907, 0.85679, 0.19906, -0.00984],
                          [0.18171, -0.17558, -0.79658, -0.00355, 0.22358]])
    fscat = atfi.const([[0.23399, 0.15044, -0.20545, 0.32825, 0.35412]] + 4 * [[0.15044, 0, 0, 0, 0]])
    b_poles = atfi.cast_complex(atfi.const([1.0, 0.5, 0.2, 0.1, 0.0]))
    f_prod = atfi.cast_complex(atfi.const([0.3, 0.2, 0.1, 0.1, 0.0]))
    mk, meta, metap = (atfi.const(m) for m in (0.493677, 0.54775, 0.95778))
    mpi = atfi.const(MPI)
    masses = [[mpi, mpi], [mk, mk], [mpi], [meta, meta], [meta, metap]]
    s = atfi.const(np.linspace(0.1, 2.5, 200))
    args = (s, m_poles, g_poles, atfi.const(-3.92637), fscat, b_poles, atfi.const(-0.07), f_prod, masses)
    default = atfd.kmatrix_lineshape(*args).numpy()
    ampgen = atfd.kmatrix_lineshape(*args, multimeson="ampgen").numpy()
    belle = atfd.kmatrix_lineshape(*args, multimeson="belle").numpy()
    np.testing.assert_array_equal(default, ampgen)
    high = s.numpy() > 1.0
    np.testing.assert_allclose(belle[high], ampgen[high], rtol=1e-12)
    assert np.max(np.abs(belle[~high] / ampgen[~high] - 1)) > 1e-3
