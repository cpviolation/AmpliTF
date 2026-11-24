# Copyright 2024 CERN for the benefit of the LHCb collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""
Lineshapes with the conventions of AmpGen (``src/Lineshapes*.cpp``,
https://github.com/GooFit/AmpGen), for models defined with AmpGen.

They differ from :mod:`amplitf.dynamics` in the normalisation: the
Blatt-Weisskopf form factor of the decay vertex is normalised at :math:`q = 0`,
the running width uses the form factor normalised at the nominal mass, and
the amplitudes include AmpGen's constant ``kFactor``.

Arguments: ``s`` is the invariant mass squared of the resonance, ``s1`` and
``s2`` those of its two daughters, ``L`` the orbital angular momentum of the
decay, ``radius`` the Blatt-Weisskopf radius in GeV^-1.
"""

import math

import tensorflow as tf

import amplitf.interface as atfi


def _t(x):
    """Python numbers to tensors with the AmpliTF floating-point type; tensors unchanged."""
    return x if isinstance(x, (tf.Tensor, tf.Variable)) else atfi.const(x)


def q2(s, s1, s2):
    """Squared breakup momentum :math:`q^2(s; s_1, s_2)` (can be negative below threshold)."""
    return (s - 2.0 * s1 - 2.0 * s2 + (s1 - s2) ** 2 / s) / 4.0


def blatt_weisskopf_norm(z, z0, L):
    """Ratio :math:`B_L^2(z)/B_L^2(z_0)` of squared Blatt-Weisskopf factors, :math:`z = q^2 r^2`."""
    if L == 0:
        return atfi.ones(z) if hasattr(z, "shape") else 1.0
    if L == 1:
        return (1.0 + z0) / (1.0 + z)
    if L == 2:
        return (z0 * z0 + 3.0 * z0 + 9.0) / (z * z + 3.0 * z + 9.0)
    if L == 3:
        return (z0 * (z0 - 15.0) ** 2 + 9.0 * (2.0 * z0 - 5.0) ** 2) / (z * (z - 15.0) ** 2 + 9.0 * (2.0 * z - 5.0) ** 2)
    raise NotImplementedError(f"Blatt-Weisskopf factor for L = {L}")


def kfactor(mass, width):
    """AmpGen's constant normalisation factor of the Breit-Wigner and Gounaris-Sakurai lineshapes."""
    mass, width = _t(mass), _t(width)
    gamma = mass * atfi.sqrt(mass * mass + width * width)
    k = (2.0 * math.sqrt(2.0) / math.pi) * mass * width * gamma / atfi.sqrt(mass * mass + gamma)
    return atfi.sqrt(k)


def running_width(s, s1, s2, mass, width, radius, L):
    r""":math:`\Gamma(s) = \Gamma_0 \frac{B_L^2(q)}{B_L^2(q_0)} \frac{m}{\sqrt{s}} \left(\frac{q}{q_0}\right)^{2L+1}`."""
    mass, width, radius = _t(mass), _t(width), _t(radius)
    q2v = q2(s, s1, s2)
    q2v = atfi.where(q2v > 0.0, q2v, atfi.zeros(q2v))
    q20 = atfi.abs(q2(mass * mass, s1, s2))
    bf = blatt_weisskopf_norm(q2v * radius * radius, q20 * radius * radius, L)
    q2r = q2v / q20
    return width * bf * mass * atfi.sqrt(q2r / s) * q2r ** L


def form_factor(s, s1, s2, radius, L):
    """Decay-vertex form factor :math:`\\sqrt{B_L^2(q)/B_L^2(0)}` (AmpGen ``FormFactor`` lineshape)."""
    z = atfi.abs(q2(s, s1, s2)) * _t(radius) ** 2
    return atfi.sqrt(blatt_weisskopf_norm(z, 0.0, L))


def form_factor_signed(s, s1, s2, radius, L):
    """As :func:`form_factor` but with :math:`q^2` not in absolute value, as used by AmpGen
    for the ``FormFactor`` lineshape of stable (head) particles."""
    z = q2(s, s1, s2) * _t(radius) ** 2
    return atfi.sqrt(blatt_weisskopf_norm(z, 0.0, L))


def breit_wigner(s, s1, s2, mass, width, radius, L):
    """AmpGen ``BW``: :math:`k F_L(q) / (m^2 - s - i m \\Gamma(s))`."""
    mass, width, radius = _t(mass), _t(width), _t(radius)
    ff = form_factor(s, s1, s2, radius, L)
    gamma = running_width(s, s1, s2, mass, width, radius, L)
    denom = atfi.complex(mass * mass - s, -mass * gamma)
    return atfi.cast_complex(kfactor(mass, width) * ff) / denom


def kfactor_complex(mass, width):
    """:func:`kfactor` for a complex width (principal square roots), as AmpGen
    evaluates it for the ``Flatte.kFactor`` lineshape."""
    mass = atfi.cast_complex(_t(mass))
    gamma = mass * atfi.sqrt(mass * mass + width * width)
    k = atfi.cast_complex(atfi.const(2.0 * math.sqrt(2.0) / math.pi)) * mass * width * gamma / atfi.sqrt(mass * mass + gamma)
    return atfi.sqrt(k)


def flatte_a0(s, mass, g2pieta, g2kk_by_g2pieta, mpi, meta, mk, kfactor=False):
    r"""AmpGen ``Flatte`` for the :math:`a_0(980)^0`: :math:`1/(m^2 - s - i m \Gamma(s))` with
    :math:`\Gamma(s) = g^2_{\pi\eta} [\rho_{\pi\eta}(s) + (g^2_{KK}/g^2_{\pi\eta}) \rho_{KK}(s)]`,
    where :math:`\rho_{KK}` is imaginary below the :math:`K^+K^-` threshold. ``kfactor=True``
    (``Flatte.kFactor``) multiplies by AmpGen's ``kFactor`` evaluated with the complex
    :math:`\Gamma(s)`. No form factor is applied.
    """
    mass, g2pieta, g2kk_by_g2pieta = _t(mass), _t(g2pieta), _t(g2kk_by_g2pieta)
    pp, pm = (mpi + meta) ** 2, (mpi - meta) ** 2
    rho_pieta = atfi.sqrt((1.0 - pm / s) * (1.0 - pp / s))
    width = atfi.cast_complex(g2pieta) * (atfi.cast_complex(rho_pieta) + atfi.cast_complex(g2kk_by_g2pieta) * phsp_two_body(s, mk, mk))
    m = atfi.cast_complex(mass)
    bw = 1.0 / (m * m - atfi.complex(atfi.const(0.0), atfi.const(1.0)) * m * width - atfi.cast_complex(s))
    return kfactor_complex(mass, width) * bw if kfactor else bw


def gounaris_sakurai(s, s1, s2, mass, width, radius, L, mpi):
    """AmpGen ``GounarisSakurai`` (running width without radius, as in AmpGen's default)."""
    mass, width, radius, mpi = _t(mass), _t(width), _t(radius), _t(mpi)
    s0 = mass * mass

    def k2(x):
        return x - 4.0 * mpi * mpi

    def k(x):
        return atfi.sqrt(atfi.where(k2(x) > 0.0, k2(x), atfi.zeros(x)))

    def log_term(x):
        return atfi.log((atfi.sqrt(x) + k(x)) / (2.0 * mpi))

    t1 = 2.0 * s0 * k2(s) * k(s) * log_term(s) / (atfi.sqrt(s) * k(s0) ** 3)
    t2 = log_term(s0) * (k2(s) * (s - 3.0 * s0) + s * (s0 - s)) / (mass * k2(s0))
    t3 = (s0 - s) / k(s0)
    m2 = s0 + width * (t1 + t2 + t3) / math.pi
    gamma = running_width(s, s1, s2, mass, width, 0.0, L)
    ff = form_factor(s, s1, s2, radius, L)
    denom = atfi.complex(m2 - s, -mass * gamma)
    return atfi.cast_complex(kfactor(mass, width) * ff) / denom


# ----------------------------------------------------------------------------
# K-matrix lineshapes (AmpGen src/Lineshapes/FOCUS.cpp and kMatrix.cpp)


def _i():
    return atfi.complex(atfi.const(0.0), atfi.const(1.0))


def complex_sqrt(x):
    r"""AmpGen ``complex_sqrt``: :math:`\sqrt{x}` for :math:`x > 0`, :math:`i\sqrt{-x}` otherwise."""
    pos = x > 0.0
    return atfi.complex(atfi.sqrt(atfi.where(pos, x, atfi.zeros(x))), atfi.sqrt(atfi.where(pos, atfi.zeros(x), -x)))


def _pol(x, coeffs):
    return sum(c * x ** i for i, c in enumerate(coeffs))


def phsp_two_body(s, m0, m1):
    r"""Two-body phase space :math:`\sqrt{1 - (m_0 + m_1)^2/s}` (AmpGen ``phsp_twoBody``)."""
    return complex_sqrt(1.0 - (m0 + m1) ** 2 / s)


def phsp_focus(s, m0, m1):
    r""":math:`\sqrt{(1 - (m_0 + m_1)^2/s)(1 - (m_0 - m_1)^2/s)}` (AmpGen ``phsp_FOCUS``)."""
    return complex_sqrt((1.0 - (m0 + m1) ** 2 / s) * (1.0 - (m0 - m1) ** 2 / s))


def focus_kpi(s, component, mk, mpi, metap):
    r"""FOCUS :math:`K\pi` S-wave (Pennington et al., as in AmpGen ``FOCUS``).

    ``component`` is ``"Kpi"`` (:math:`I = 1/2`, :math:`K\pi` row of the coupled
    :math:`K\pi`-:math:`K\eta'` system), ``"KEta"`` or ``"I32"`` (:math:`I = 3/2`).
    ``mk``, ``mpi``, ``metap`` are the :math:`K^+`, :math:`\pi^+` and :math:`\eta'` masses.
    """
    i = _i()
    snorm = mk * mk + mpi * mpi
    x = s / snorm - 1.0
    rho1, rho2 = phsp_focus(s, mk, mpi), phsp_focus(s, mk, metap)
    if component == "I32":
        k32 = atfi.cast_complex((s - 0.27) / snorm * _pol(x, [-0.22147, 0.026637, -0.00092057]))
        return 1.0 / (1.0 - i * k32 * rho1)
    adler = (s - 0.23) / snorm
    g, sp = (0.31072, -0.02323), 1.7919
    k11 = atfi.cast_complex(adler * (g[0] * g[0] / (sp - s) + _pol(x, [0.79299, -0.15099, 0.00811])))
    k12 = atfi.cast_complex(adler * (g[0] * g[1] / (sp - s) + _pol(x, [0.15040, -0.038266, 0.0022596])))
    k22 = atfi.cast_complex(adler * (g[1] * g[1] / (sp - s) + _pol(x, [0.17054, -0.0219, 0.00085655])))
    det = k11 * k22 - k12 * k12
    delta = 1.0 - rho1 * rho2 * det - i * (rho1 * k11 + rho2 * k22)
    if component == "Kpi":
        return (k11 - i * rho2 * det) / delta
    if component == "KEta":
        return k12 / delta
    raise ValueError(f"Unknown FOCUS component '{component}'")


# Masses hard-coded in AmpGen's kMatrix lineshape
KMATRIX_MPI, KMATRIX_MK, KMATRIX_META, KMATRIX_METAP = 0.139570, 0.493677, 0.547862, 0.967780
KMATRIX_CHANNELS = ("pipi", "KK", "4pi", "EtaEta", "EtapEta")


def _phsp_four_pi(s):
    poly = atfi.cast_complex(_pol(s, [0.00051, -0.01933, 0.13851, -0.20840, -0.29744, 0.13655, 1.07885]))
    return tf.where(s > 1.0, phsp_two_body(s, 2 * KMATRIX_MPI, 2 * KMATRIX_MPI), poly)


def kmatrix_ampgen_propagator(s, pole_masses, pole_couplings, f_scatt, s0_scatt, sA0, sA, imaginary_mass=1e-6):
    r"""Propagator :math:`F = (1 - i K \rho)^{-1}` of AmpGen's 5-channel :math:`\pi\pi` S-wave
    K-matrix (channels :math:`\pi\pi, KK, 4\pi, \eta\eta, \eta\eta'`), the Anisovich-Sarantsev
    parametrisation as coded in AmpGen ``kMatrix``.

    ``pole_masses`` (5) and ``pole_couplings`` (5 x 5, pole x channel) are AmpGen's
    ``IS_p<n>_mass`` and ``IS_p<n>_<channel>``; ``f_scatt`` (5), ``s0_scatt``, ``sA0`` and ``sA``
    its non-resonant and Adler-zero parameters. Pole positions are
    :math:`m^2 + i\,\epsilon` with ``imaginary_mass`` :math:`= \epsilon` (``kMatrix::fp``).
    Returns ``(F, pole_s)``: ``F`` of shape ``[N, 5, 5]`` and the complex pole positions.
    """
    n = len(KMATRIX_CHANNELS)
    sc = atfi.cast_complex(s)
    pole_s = [atfi.complex(_t(m) * _t(m), atfi.const(imaginary_mass)) for m in pole_masses]
    adler = (1.0 - sA0) * (s - sA * KMATRIX_MPI ** 2 / 2.0) / (s - sA0)
    scatt = (1.0 - s0_scatt) / (s - s0_scatt)
    rows = []
    for a in range(n):
        row = []
        for b in range(n):
            k = sum(atfi.cast_complex(_t(g[a]) * _t(g[b])) / (ps - sc) for g, ps in zip(pole_couplings, pole_s))
            if a == 0 or b == 0:
                k = k + atfi.cast_complex(_t(f_scatt[max(a, b)]) * scatt)
            row.append(k * atfi.cast_complex(adler))
        rows.append(row)
    rho = [phsp_two_body(s, KMATRIX_MPI, KMATRIX_MPI), phsp_two_body(s, KMATRIX_MK, KMATRIX_MK), _phsp_four_pi(s),
           phsp_two_body(s, KMATRIX_META, KMATRIX_META), phsp_two_body(s, KMATRIX_META, KMATRIX_METAP)]
    one, zero = atfi.cast_complex(atfi.ones(s)), atfi.cast_complex(atfi.zeros(s))
    t = tf.stack([tf.stack([(one if a == b else zero) - _i() * rows[a][b] * rho[b] for b in range(n)], axis=-1)
                  for a in range(n)], axis=-2)
    return tf.linalg.inv(t), pole_s


def kmatrix_ampgen(s, term, index, pole_masses, pole_couplings, f_scatt, s0_scatt, sA0, sA, s0_prod,
                   imaginary_mass=1e-6):
    r"""A production term of AmpGen's ``kMatrix`` lineshape (``kMatrix.<term>.<index>``):

    - ``pole`` / ``poleKK``: :math:`\sum_j F_{cj} g_{\alpha j} / (s_\alpha - s)` for pole :math:`\alpha` = ``index``;
    - ``prod`` / ``prodKK``: :math:`F_{cj} (1 - s_0^{prod}) / (s - s_0^{prod})` for channel :math:`j` = ``index``;

    with :math:`c = \pi\pi` (``pole``, ``prod``) or :math:`KK` (``poleKK``, ``prodKK``).
    See :func:`kmatrix_ampgen_propagator` for the other arguments.
    """
    F, pole_s = kmatrix_ampgen_propagator(s, pole_masses, pole_couplings, f_scatt, s0_scatt, sA0, sA, imaginary_mass)
    row = {"pole": 0, "prod": 0, "poleKK": 1, "prodKK": 1}[term]
    if term.startswith("pole"):
        g = pole_couplings[index]
        m = sum(F[..., row, j] * atfi.cast_complex(_t(g[j]) * atfi.ones(s)) for j in range(len(KMATRIX_CHANNELS)))
        return m / (pole_s[index] - atfi.cast_complex(s))
    return F[..., row, index] * atfi.cast_complex((1.0 - s0_prod) / (s - s0_prod))


# ----------------------------------------------------------------------------
# Spline lineshapes (AmpGen src/Spline.cpp and src/Lineshapes/SplineShapes.cpp)

def spline_curvatures(values, xmin, xmax):
    """Second derivatives at the knots of AmpGen's natural cubic spline through ``values``
    (uniform knots from ``xmin`` to ``xmax``; zero curvature at both ends)."""
    import numpy as np

    y = np.asarray(values, dtype=float)
    n = len(y)
    h = (xmax - xmin) / (n - 1)
    m = np.diag(np.full(n - 2, 4.0)) + np.diag(np.ones(n - 3), 1) + np.diag(np.ones(n - 3), -1)
    curv = np.zeros(n)
    curv[1:-1] = np.linalg.solve(m, 6.0 * (y[2:] - 2.0 * y[1:-1] + y[:-2]) / h ** 2)
    return curv


def spline(x, values, xmin, xmax):
    """AmpGen's ``Spline``: natural cubic spline through ``values`` at uniform knots in
    [``xmin``, ``xmax``]. Outside the open interval it returns the value of the last knot
    (also below ``xmin``), as AmpGen does."""
    import numpy as np

    n = len(values)
    h = (xmax - xmin) / (n - 1)
    y = atfi.const(np.asarray(values, dtype=float))
    c = atfi.const(spline_curvatures(values, xmin, xmax))
    inside = (x > xmin) & (x < xmax)
    u = atfi.where(inside, x - xmin, atfi.zeros(x))
    b = tf.cast(tf.floor(u / h), tf.int32)
    b = tf.minimum(b, n - 2)
    dx = u - tf.cast(b, u.dtype) * h
    y0, y1, c0, c1 = (tf.gather(t, i) for t in (y, c) for i in (b, b + 1))
    val = y0 + ((y1 - y0) / h - (c1 + 2.0 * c0) * h / 6.0) * dx + c0 * dx * dx / 2.0 + dx ** 3 * (c1 - c0) / (6.0 * h)
    return atfi.where(inside, val, atfi.ones(x) * float(values[-1]))


def gspline(s, s1, s2, mass, width, radius, L, values, xmin, xmax, form_factor="BlattWeisskopf"):
    r"""AmpGen ``GSpline``: Breit-Wigner with the running width given by a spline,
    :math:`k F(q) / (m^2 - s - i m \Gamma_0 G(s)/G(m^2))`, where :math:`G` is the
    spline through ``values`` (AmpGen ``<particle>::Spline::Gamma::<i>``) on
    [``xmin``, ``xmax``] in :math:`s`.

    ``form_factor`` selects :math:`F(q)`: ``"BlattWeisskopf"`` (default,
    :math:`\sqrt{B_L^2(q)/B_L^2(0)}`) or ``"EFF"`` (``GSpline.EFF``,
    :math:`e^{-q^2 r^2/2}`), with :math:`q^2 = |q^2(s; s_1, s_2)|`.
    """
    mass, width, radius = _t(mass), _t(width), _t(radius)
    q2v = atfi.abs(q2(s, s1, s2))
    if form_factor == "EFF":
        ff = atfi.exp(-q2v * radius * radius / 2.0)
    elif form_factor == "BlattWeisskopf":
        ff = atfi.sqrt(blatt_weisskopf_norm(q2v * radius * radius, 0.0, L))
    else:
        raise ValueError(f"Unknown GSpline form factor '{form_factor}'")
    shape = spline(s, values, xmin, xmax)
    norm = spline(mass * mass * atfi.ones(s), values, xmin, xmax)
    gamma = width * shape / norm
    return atfi.cast_complex(kfactor(mass, width) * ff) / atfi.complex(mass * mass - s, -mass * gamma)
