"""Mixing formulas against an explicit time evolution of the mass eigenstates, and
decay-time sampling with DecayTimePhaseSpace."""

import numpy as np
import pytest
import tensorflow as tf

import amplitf.interface as atfi
import amplitf.mixing as atfm
from amplitf.phasespace.decaytime_phasespace import DecayTimePhaseSpace

X, Y, TAU = 0.03, 0.02, 1.0          # exaggerated so that sign errors are visible
QOP = 0.9 * np.exp(0.3j)


def c(z):
    return tf.constant(z, dtype=atfi.ctype())


def explicit(t, af, abf):
    """|<f|H|D0(t)>|^2 and |<f|H|D0bar(t)>|^2, with |D1,2> = p|D0> +- q|D0bar>,
    x = (m1-m2)/Gamma, y = (Gamma1-Gamma2)/(2 Gamma)."""
    gamma = 1.0 / TAU
    p = 1.0 / np.sqrt(1 + abs(QOP) ** 2)
    q = QOP * p
    e1 = np.exp(-1j * X * gamma / 2 * t - gamma * (1 + Y) * t / 2)
    e2 = np.exp(+1j * X * gamma / 2 * t - gamma * (1 - Y) * t / 2)
    f1, f2 = p * af + q * abf, p * af - q * abf
    return np.abs((e1 * f1 + e2 * f2) / (2 * p)) ** 2, np.abs((e1 * f1 - e2 * f2) / (2 * q)) ** 2


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(1)
    n = 1000
    af = rng.normal(size=n) + 1j * rng.normal(size=n)
    abf = rng.normal(size=n) + 1j * rng.normal(size=n)
    t = rng.exponential(TAU, n)
    return af, abf, t


def evolution(t, relative=False):
    t, y, x, tau = atfi.const(t), atfi.const(Y), atfi.const(X), atfi.const(TAU)
    return (atfm.psip(t, y, tau, relative), atfm.psim(t, y, tau, relative),
            atfm.psii(t, x, tau, relative))


def test_d0_and_d0bar(inputs):
    af, abf, t = inputs
    d0, d0bar = explicit(t, af, abf)
    dens = atfm.mixing_density(c(af), c(abf), c(QOP), *evolution(t)).numpy()
    np.testing.assert_allclose(dens, d0, rtol=1e-12)
    # initial D0bar: exchange A_f <-> Abar_f and q/p -> p/q, via flavor=False
    dens_bar = atfm.mixing_density(c(abf), c(af), c(QOP), *evolution(t), flavor=False).numpy()
    np.testing.assert_allclose(dens_bar, d0bar, rtol=1e-12)
    # equivalent to passing p/q explicitly
    explicit_pq = atfm.mixing_density(c(abf), c(af), c(1 / QOP), *evolution(t)).numpy()
    np.testing.assert_allclose(dens_bar, explicit_pq, rtol=1e-12)


def test_relative_to_decay(inputs):
    af, abf, t = inputs
    full = atfm.mixing_density(c(af), c(abf), c(QOP), *evolution(t)).numpy()
    rel = atfm.mixing_density(c(af), c(abf), c(QOP), *evolution(t, relative=True)).numpy()
    np.testing.assert_allclose(rel * np.exp(-t / TAU), full, rtol=1e-12)


def test_decay_time_density():
    tphsp = DecayTimePhaseSpace(atfi.const(TAU))
    sample = tphsp.uniform_sample(400000).numpy()
    assert sample[:, 0].mean() == pytest.approx(TAU, rel=0.01)
    t = np.linspace(0.0, 5.0, 11)
    np.testing.assert_allclose(tphsp.density(atfi.const(t[:, None])).numpy(), np.exp(-t / TAU) / TAU)


def test_accept_reject_with_relative_density():
    """Accept-reject on DecayTimePhaseSpace samples with the relative density gives the true PDF."""
    tphsp = DecayTimePhaseSpace(atfi.const(TAU))
    t = tphsp.uniform_sample(400000).numpy()[:, 0]
    y = 0.2  # only the |A1|^2 term: true PDF ~ exp(-(1+y) t / tau)
    w = atfm.psip(atfi.const(t), atfi.const(y), atfi.const(TAU), True).numpy()
    keep = np.random.default_rng(2).uniform(0, w.max(), len(w)) < w
    assert t[keep].mean() == pytest.approx(TAU / (1 + y), rel=0.01)


def test_single_precision():
    atfi.set_single_precision()
    try:
        r = atfm.psii(atfi.const([0.5, 1.0]), atfi.const(X), atfi.const(TAU)).numpy()
        assert r.dtype == np.complex64
    finally:
        atfi.set_double_precision()
