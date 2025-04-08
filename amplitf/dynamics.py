# Copyright 2017 CERN for the benefit of the LHCb collaboration
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

import tensorflow as tf
import amplitf.interface as atfi
import amplitf.kinematics as atfk


@atfi.function
def density(ampl):
    """density for a complex amplitude: :math:`|ampl|^2`

    Args:
        ampl (complex): the amplitude

    Returns:
        float: density for the amplitude
    """
    return abs(ampl) ** 2


@atfi.function
def polar(a, ph):
    """Create a complex number from polar coordinates

    Args:
        a (float): magnitude
        ph (float): phase

    Returns:
        complex: complex number from magnitude and phase
    """
    return complex(a * cos(ph), a * sin(ph))


@atfi.function
def argument(c):
    """Return argument (phase) of a complex number :math:`atan2(imag(c), real(c))`

    Args:
        c (complex): complex number

    Returns:
        float: argument of the complex number
    """
    return atan2(imag(c), real(c))


@atfi.function
def helicity_amplitude(x, spin):
    r"""Helicity amplitude for a resonance in scalar-scalar state
        
        - spin 0: :math:`1`
        - spin 1: :math:`x`
        - spin 2: :math:`(3x^2-1)/2`
        - spin 3: :math:`(5x^3-3x)/2`
        - spin 4: :math:`(35x^4-30x^2+3)/8`
        - spin < 0 or > 4: None

    Args:
        x (float): cos(helicity angle)
        spin (int): spin of the resonance

    Returns:
        complex: the helicity amplitude
    """
    if spin == 0:
        return atfi.complex(atfi.const(1.0), atfi.const(0.0))
    if spin == 1:
        return atfi.complex(x, atfi.const(0.0))
    if spin == 2:
        return atfi.complex((3.0 * x ** 2 - 1.0) / 2.0, atfi.const(0.0))
    if spin == 3:
        return atfi.complex((5.0 * x ** 3 - 3.0 * x) / 2.0, atfi.const(0.0))
    if spin == 4:
        return atfi.complex(
            (35.0 * x ** 4 - 30.0 * x ** 2 + 3.0) / 8.0, atfi.const(0.0)
        )
    return None


@atfi.function
def relativistic_breit_wigner(m2, mres, wres):
    r"""Relativistic Breit-Wigner

        .. math::

            BW(m^2) = \frac{1}{m_{res}^2 - m^2 - i m_{res} \Gamma}


    Args:
        m2 (float): invariant mass squared of the resonating system
        mres (float): mass of the resonance
        wres (float): width of the resonance

    Returns:
        complex: relativistic Breit-Wigner amplitude
    """
    if wres.dtype is atfi.ctype():
        return tf.math.reciprocal(
            atfi.cast_complex(mres * mres - m2)
            - atfi.complex(atfi.const(0.0), mres) * wres
        )
    if wres.dtype is atfi.fptype():
        return tf.math.reciprocal(atfi.complex(mres * mres - m2, -mres * wres))
    return None


@atfi.function
def blatt_weisskopf_ff(q, q0, d, l):
    r"""Blatt-Weisskopf form factor for intermediate resonance

        .. math::

            BWFF(q) = \sqrt{\frac{H_l(z_0)}{H_l(z)}}

    with :math:`z = q d` and :math:`z_0 = q_0 d` and :math:`H_l(z)` (Hankel function) defined as

        - :math:`H_0(z) = 1`
        - :math:`H_1(z) = 1 + z^2`
        - :math:`H_2(z) = 9 + z^2 (3 + z^2)`
        - :math:`H_3(z) = 225 + z^2 (45 + z^2 (6 + z^2))`
        - :math:`H_4(z) = 11025 + z^2 (1575 + z^2 (135 + z^2 (10 + z^2)))`

    Args:
        q (float): q-value at the invariant mass of the system
        q0 (float): q-value at the resonance
        d (float): barrier radius
        l (int): the orbital angular momentum of the resonance

    Returns:
        float: Blatt-Weisskopf form factor
    """
    z = q * d
    z0 = q0 * d

    def hankel1(x):
        if l == 0:
            return atfi.const(1.0)
        if l == 1:
            return 1 + x * x
        if l == 2:
            x2 = x * x
            return 9 + x2 * (3.0 + x2)
        if l == 3:
            x2 = x * x
            return 225 + x2 * (45 + x2 * (6 + x2))
        if l == 4:
            x2 = x * x
            return 11025.0 + x2 * (1575.0 + x2 * (135.0 + x2 * (10.0 + x2)))

    return atfi.sqrt(hankel1(z0) / hankel1(z))


@atfi.function
def blatt_weisskopf_ff_squared(q_squared, d, l_orbit):
    r"""Blatt-Weisskopf form factor squared for intermediate resonance (:math:`BWFF2(q)`)

        - :math:`l = 0`: :math:`1`
        - :math:`l = 1`: :math:`\frac{2z}{z+1}`
        - :math:`l = 2`: :math:`\frac{13z^2}{(z-3)^2 + 9z}`
        - :math:`l = 3`: :math:`\frac{277z^3}{z(z-15)^2 + 9(2z-5)^2}`
        - :math:`l = 4`: :math:`\frac{12746z^4}{(z^2 - 45z + 105)^2 + 25z(2z-21)^2}`

        with :math:`z = q^2 d^2`

    Args:
        q_squared (float): q-value squared at the invariant mass of the system
        d (float): barrier radius
        l_orbit (int): the orbital angular momentum of the resonance

    Returns:
        float: the squared Blatt-Weisskopf form factor
    """
    z = q_squared * d * d

    def _bw_ff_squared(x):
        if l_orbit == 0:
            return atfi.const(1.0)
        if l_orbit == 1:
            return (2 * x) / (x + 1)
        if l_orbit == 2:
            return (13 * x * x) / ((x - 3) * (x - 3) + 9 * x)
        if l_orbit == 3:
            return (277 * x * x * x) / (
                x * (x - 15) * (x - 15) + 9 * (2 * x - 5) * (2 * x - 5)
            )
        if l_orbit == 4:
            return (12746 * x * x * x * x) / (
                (x * x - 45 * x + 105) * (x * x - 45 * x + 105)
                + 25 * x * (2 * x - 21) * (2 * x - 21)
            )

    return _bw_ff_squared(z)


@atfi.function
def mass_dependent_width(m, m0, gamma0, p, p0, ff, l):
    r"""Mass dependent width for Breit-Wigner amplitude

        - :math:`l = 0`: :math:`\Gamma = \Gamma_0 \frac{p}{p_0} \frac{m_0}{m} |FF|^2`
        - :math:`l = 1`: :math:`\Gamma = \Gamma_0 \left(\frac{p}{p_0}\right)^3 \frac{m_0}{m} |FF|^2`
        - :math:`l = 2`: :math:`\Gamma = \Gamma_0 \left(\frac{p}{p_0}\right)^5 \frac{m_0}{m} |FF|^2`
        - :math:`l \geq 3`: :math:`\Gamma = \Gamma_0 \left(\frac{p}{p_0}\right)^{2l+1} \frac{m_0}{m} |FF|^2`

    Args:
        m (float): invariant mass of the system
        m0 (float): resonance mass
        gamma0 (float): resonance width
        p (float): momentum of the system
        p0 (float): momentum of the resonance
        ff (float): form factor
        l (int): orbital angular momentum of the resonance

    Returns:
        float: mass dependent width
    """
    if l == 0:
        return gamma0 * (p / p0) * (m0 / m) * (ff * ff)
    if l == 1:
        return gamma0 * ((p / p0) ** 3) * (m0 / m) * (ff * ff)
    if l == 2:
        return gamma0 * ((p / p0) ** 5) * (m0 / m) * (ff * ff)
    if l >= 3:
        return gamma0 * ((p / p0) ** (2 * l + 1)) * (m0 / m) * (ff ** 2)


@atfi.function
def orbital_barrier_factor(p, p0, l):
    r"""Orbital barrier factor :math:`B_l` for the resonance

        - :math:`l = 0`: :math:`1`
        - :math:`l = 1`: :math:`\frac{p}{p_0}`
        - :math:`l \geq 2`: :math:`\left(\frac{p}{p_0}\right)^l`

    n.b.: when :math:`p_0 = 0` (resonance pole outside phase space), the barrier factor is 1

    Args:
        p (float): momentum of the system
        p0 (float): momentum of the system assuming the resonance mass
        l (int): orbital angular momentum of the resonance

    Returns:
        float: orbital barrier factor
    """
    if l == 0:
        return atfi.ones(p)
    return tf.where(p0 == 0, atfi.const(1), (p / p0) ** l)


@atfi.function
def breit_wigner_lineshape(
    m2,
    m0,
    gamma0,
    ma,
    mb,
    mc,
    md,
    dr,
    dd,
    lr,
    ld,
    barrier_factor=True,
    ma0=None,
    md0=None,
):
    r"""Breit-Wigner amplitude with Blatt-Weisskopf form factors, mass-dependent width and orbital barriers

        .. math::

            BW(m^2) = \frac{1}{m_{res}^2 - m^2 - i m_{res} \Gamma(m, m_{res}, \Gamma_{res}, p, p_0, FF_r, l_r)} FF_r FF_d

            
        if barrier_factor is True, the orbital barrier factors are included in the form factor

        .. math::

            BW(m^2) = \frac{1}{m_{res}^2 - m^2 - i m_{res} \Gamma(m, m_{res}, \Gamma_{res}, p, p_0, FF_r, l_r)} FF_r FF_d B_r B_d

        
        where
            - :math:`\Gamma(m, m_{res}, \Gamma_{res}, p, p_0, FF_r, l_r)` is the mass-dependent width
            - :math:`FF_r = BWFF(p, p_0, d_r, l_r)` is the Blatt-Weisskopf form factor for the resonance
            - :math:`FF_d = BWFF(p, p_0, d_d, l_d)` is the Blatt-Weisskopf form factor for the decay
            - :math:`B_r = B_l(p, p_0, l_r)` is the orbital barrier factor for the resonance
            - :math:`B_d = B_l(q, q_0, l_d)` is the orbital barrier factor for the decay

    Args:
        m2 (float): invariant mass squared of the system
        m0 (float): resonance mass
        gamma0 (float): resonance width
        ma (float): mass of particle a
        mb (float): mass of particle b
        mc (float): mass of the other particle (particle c)
        md (float): mass of the decaying particle
        dr (float): barrier radius for the resonance
        dd (float): barrier radius for the decay
        lr (int): orbital angular momentum of the resonance
        ld (int): orbital angular momentum of the decay
        barrier_factor (bool, optional): multiplies the form factor for the barrier factors. Defaults to True.
        ma0 (float, optional): alternative value for the mass of the decaying particle to calculate the momentum of the system assuming the resonance mass. Defaults to None.
        md0 (float, optional): alternative value for the mass of the decaying particle to calculate the q-value of the system assuming the resonance mass. Defaults to None.

    Returns:
        complex: the Breit-Wigner amplitude
    """
    m = atfi.sqrt(m2)
    q = atfk.two_body_momentum(md, m, mc)
    q0 = atfk.two_body_momentum(md if md0 is None else md0, m0, mc)
    p = atfk.two_body_momentum(m, ma, mb)
    p0 = atfk.two_body_momentum(m0, ma if ma0 is None else ma0, mb)
    ffr = blatt_weisskopf_ff(p, p0, dr, lr)
    ffd = blatt_weisskopf_ff(q, q0, dd, ld)
    width = mass_dependent_width(m, m0, gamma0, p, p0, ffr, lr)
    bw = relativistic_breit_wigner(m2, m0, width)
    ff = ffr * ffd
    if barrier_factor:
        b1 = orbital_barrier_factor(p, p0, lr)
        b2 = orbital_barrier_factor(q, q0, ld)
        ff *= b1 * b2
    return bw * atfi.complex(ff, atfi.const(0.0))


@atfi.function
def breit_wigner_decay_lineshape(m2, m0, gamma0, ma, mb, meson_radius, l_orbit):
    r"""Breit-Wigner amplitude with Blatt-Weisskopf form factor for the decay products,
    mass-dependent width and orbital barriers

    .. math::

        BW(m^2) = \frac{1}{m_{res}^2 - m^2 - i m_{res} \Gamma} m_{res} \Gamma_0 \sqrt{FF^2}

    with

    .. math::

        \Gamma = \Gamma_0 \frac{m_{res}}{m} \frac{BWFF2(q^2)}{BWFF2(q_0^2)} \sqrt{\frac{q^2}{q_0^2}}

    Note: This function does not include the production form factor.

    Args:
        m2 (float): invariant mass squared of the system
        m0 (float): resonance mass
        gamma0 (float): resonance width
        ma (float): mass of particle a
        mb (float): mass of particle b
        meson_radius (float): barrier radius for the resonance
        l_orbit (int): orbital angular momentum of the resonance

    Returns:
        complex: Breit-Wigner amplitude
    """
    inv_mass = atfi.sqrt(m2)
    q_squared = atfk.two_body_momentum_squared(inv_mass, ma, mb)
    q0_squared = atfk.two_body_momentum_squared(m0, ma, mb)
    ff2 = blatt_weisskopf_ff_squared(q_squared, meson_radius, l_orbit)
    ff02 = blatt_weisskopf_ff_squared(q0_squared, meson_radius, l_orbit)
    width = gamma0 * (m0 / inv_mass) * (ff2 / ff02)
    # So far its all in float64,
    # but for the sqrt operation it has to be converted to complex
    width = atfi.complex(width, atfi.const(0.0)) * atfi.sqrt(
        atfi.complex(
            (q_squared / q0_squared),
            atfi.const(0.0),
        )
    )
    return relativistic_breit_wigner(m2, m0, width) * atfi.complex(
        m0 * gamma0 * atfi.sqrt(ff2), atfi.const(0.0)
    )


@atfi.function
def subthreshold_breit_wigner_lineshape(
    m2, m0, gamma0, ma, mb, mc, md, dr, dd, lr, ld, barrier_factor=True
):
    r"""Breit-Wigner amplitude (with the mass under kinematic threshold)
    with Blatt-Weisskopf form factors, mass-dependent width and orbital barriers

    .. math::

        BW(m^2) = \frac{1}{m_{res}^2 - m^2 - i m_{res} \Gamma(m, m_{res}, \Gamma_{res}, p, p_0, FF_r, l_r)} FF_r FF_d

            
    if barrier_factor is True, the orbital barrier factors are included in the form factor

    .. math::

        BW(m^2) = \frac{1}{m_{res}^2 - m^2 - i m_{res} \Gamma(m, m_{res}, \Gamma_{res}, p, p_0, FF_r, l_r)} FF_r FF_d B_r B_d

        
    where
        - :math:`\Gamma(m, m_{res}, \Gamma_{res}, p, p_0, FF_r, l_r)` is the mass-dependent width
        - :math:`FF_r = BWFF(p, p_0, d_r, l_r)` is the Blatt-Weisskopf form factor for the resonance
        - :math:`FF_d = BWFF(p, p_0, d_d, l_d)` is the Blatt-Weisskopf form factor for the decay
        - :math:`B_r = B_l(p, p_0, l_r)` is the orbital barrier factor for the resonance
        - :math:`B_d = B_l(q, q_0, l_d)` is the orbital barrier factor for the decay

    Args:
        m2 (float): invariant mass squared of the system
        m0 (float): resonance mass
        gamma0 (float): resonance width
        ma (float): mass of particle a
        mb (float): mass of particle b
        mc (float): mass of the other particle (particle c)
        md (float): mass of the decaying particle
        dr (float): barrier radius for the resonance
        dd (float): barrier radius for the decay
        lr (int): orbital angular momentum of the resonance
        ld (int): orbital angular momentum of the decay
        barrier_factor (bool, optional): multiplies the form factor for the barrier factors. Defaults to True.

    Returns:
        complex: Breit-Wigner amplitude
    """
    m = atfi.sqrt(m2)
    mmin = ma + mb
    mmax = md - mc
    tanhterm = atfi.tanh((m0 - ((mmin + mmax) / 2.0)) / (mmax - mmin))
    m0eff = mmin + (mmax - mmin) * (1.0 + tanhterm) / 2.0
    q = atfk.two_body_momentum(md, m, mc)
    q0 = atfk.two_body_momentum(md, m0eff, mc)
    p = atfk.two_body_momentum(m, ma, mb)
    p0 = atfk.two_body_momentum(m0eff, ma, mb)
    ffr = blatt_weisskopf_ff(p, p0, dr, lr)
    ffd = blatt_weisskopf_ff(q, q0, dd, ld)
    width = mass_dependent_width(m, m0, gamma0, p, p0, ffr, lr)
    bw = relativistic_breit_wigner(m2, m0, width)
    ff = ffr * ffd
    if barrier_factor:
        b1 = orbital_barrier_factor(p, p0, lr)
        b2 = orbital_barrier_factor(q, q0, ld)
        ff *= b1 * b2
    return bw * atfi.complex(ff, atfi.const(0.0))


@atfi.function
def exponential_nonresonant_lineshape(
    m2, m0, alpha, ma, mb, mc, md, lr, ld, barrierFactor=True
):
    r"""Exponential nonresonant amplitude with orbital barriers

    .. math::

        \text{NR}_{\text{exp}}(m^2) = \exp(-\alpha(m^2 - m_0^2))

    If `barrier_factor` is `True`, the orbital barrier factors are included in the form factor

    .. math::

        \text{NR}_{\text{exp}}(m^2) = \exp(-\alpha(m^2 - m_0^2)) B_r B_d

    Args:
        m2 (float): invariant mass squared of the system
        m0 (float): resonance mass
        alpha (float): slope of the exponential
        ma (float): mass of particle a
        mb (float): mass of particle b
        mc (float): mass of the other particle (particle c)
        md (float): mass of the decaying particle
        lr (int): orbital angular momentum of the resonance
        ld (int): orbital angular momentum of the decay
        barrier_factor (bool, optional): multiplies the form factor for the barrier factors. Defaults to True.

    Returns:
        complex: Exponential non-resonant lineshape amplitude
    """
    if barrierFactor:
        m = atfi.sqrt(m2)
        q = atfk.two_body_momentum(md, m, mc)
        q0 = atfk.two_body_momentum(md, m0, mc)
        p = atfk.two_body_momentum(m, ma, mb)
        p0 = atfk.two_body_momentum(m0, ma, mb)
        b1 = orbital_barrier_factor(p, p0, lr)
        b2 = orbital_barrier_factor(q, q0, ld)
        return atfi.complex(
            b1 * b2 * atfi.exp(-alpha * (m2 - m0 ** 2)), atfi.const(0.0)
        )
    else:
        return atfi.complex(atfi.exp(-alpha * (m2 - m0 ** 2)), atfi.const(0.0))


@atfi.function
def polynomial_nonresonant_lineshape(
    m2, m0, coeffs, ma, mb, mc, md, lr, ld, barrierFactor=True
):
    r"""Second order polynomial nonresonant amplitude with orbital barriers

    .. math::

        \text{NR}_{\text{poly}}(m^2) = a_0 + a_1(m - m_0) + a_2(m - m_0)^2

    If `barrier_factor` is `True`, the orbital barrier factors are included in the form factor

    .. math::

        \text{NR}_{\text{poly}}(m^2) = a_0 + a_1(m - m_0) + a_2(m - m_0)^2 B_r B_d    

    Args:
        m2 (float): invariant mass squared of the system
        m0 (float): resonance mass
        coeffs (list): list of atfi.complex polynomial coefficients [:math:`a0`, :math:`a1`, :math:`a2`]
        ma (float): mass of particle a
        mb (float): mass of particle b
        mc (float): mass of the other particle (particle c)
        md (float): mass of the decaying particle
        lr (int): orbital angular momentum of the resonance
        ld (int): orbital angular momentum of the decay
        barrier_factor (bool, optional): multiplies the form factor for the barrier factors. Defaults to True.

    Returns:
        complex: Second order polynomial non-resonant lineshape amplitude
    """
    def poly(x, cs):
        return (
            cs[0]
            + cs[1] * atfi.complex(x, atfi.const(0.0))
            + cs[2] * atfi.complex(x ** 2, atfi.const(0.0))
        )

    m = atfi.sqrt(m2)
    if barrierFactor:
        q = atfk.two_body_momentum(md, m, mc)
        q0 = atfk.two_body_momentum(md, m0, mc)
        p = atfk.two_body_momentum(m, ma, mb)
        p0 = atfk.two_body_momentum(m0, ma, mb)
        b1 = orbital_barrier_factor(p, p0, lr)
        b2 = orbital_barrier_factor(q, q0, ld)
        return poly(m - m0, coeffs) * atfi.complex(b1 * b2, atfi.const(0.0))
    else:
        return poly(m - m0, coeffs)


@atfi.function
def gounaris_sakurai_lineshape(s, m, gamma, m_pi):
    r"""Gounaris-Sakurai shape for :math:`\rho\to\pi\pi`

    .. math::

        \text{GS}(s) = \frac{m_0^2 - m^2 +f +i m \Gamma_0}{\left(m_0^2 - m^2 + f \right)^2 + m^2 \Gamma_0^2}
        
    with

    .. math::

        \begin{align*}
        f &= \frac{\Gamma_0 m_0^2}{p_0^3} \left(p_{\pi\pi}^2 (h(m^2) - h(m_0^2)) - p_0^2 (m^2 - m_0^2) \left.\frac{d h(m^2)}{d m}\right|_{m_0^2}\right)\\
        h(m) &= \frac{2}{\pi} \frac{p(m)}{\sqrt{m}} \log\left(\frac{m + 2 p(m)}{2 m_{\pi}}\right)\\
        p(m) &= \frac{m^2 - 4 m_{\pi}^2}{4}\\
        p_{\pi\pi}^2 &= p^2(m)\\
        p_0^2 &= p^2(m_0^2)\\
        \end{align*}
       

    Args:
        s (float): invariant mass squared of the system
        m (float): resonance mass
        gamma (float): resonance width
        m_pi (float): pion mass

    Returns:
        complex: Gounaris-Sakurai amplitude
    """
    m2 = m * m
    m_pi2 = m_pi * m_pi
    ss = atfi.sqrt(s)

    ppi2 = (s - 4.0 * m_pi2) / 4.0
    p02 = (m2 - 4.0 * m_pi2) / 4.0
    p0 = atfi.sqrt(p02)
    ppi = atfi.sqrt(ppi2)

    hs = 2.0 * ppi / atfi.pi() / ss * atfi.log((ss + 2.0 * ppi) / 2.0 / m_pi)
    hm = 2.0 * p0 / atfi.pi() / m * atfi.log((m + 2.0 * ppi) / 2.0 / m_pi)

    dhdq = hm * (1.0 / 8.0 / p02 - 1.0 / 2.0 / m2) + 1.0 / 2.0 / atfi.pi() / m2
    f = gamma * m2 / (p0 ** 3) * (ppi2 * (hs - hm) - p02 * (s - m2) * dhdq)

    gamma_s = gamma * m2 * (ppi ** 3) / s / (p0 ** 3)

    dr = m2 - s + f
    di = ss * gamma_s

    r = dr / (dr ** 2 + di ** 2)
    i = di / (dr ** 2 + di ** 2)

    return atfi.complex(r, i)


@atfi.function
def flatte_lineshape(s, m, g1, g2, ma1, mb1, ma2, mb2):
    r"""Flatté line shape

    .. math::

        F(m^2) = \frac{1}{m_{res}^2 - m^2 - i m_{res} \Gamma}

    with

    .. math::

        \begin{align*}
        \Gamma &= \frac{g_1^2 \rho_1 + g_2^2 \rho_2}{m}\\
        \rho_i &= \frac{2 p_i(m)}{m}
        \end{align*}

    where :math:`p_i(m)` is the momentum of two-body decay products R->AB in the R rest frame for the decay channel :math:`i`

    Args:
        s (float): invariant mass squared of the system
        m (float): resonance mass
        g1 (float): coupling to ma1, mb1
        g2 (float): coupling to ma2, mb2
        ma1 (float): mass of particle 1 in channel 1
        mb1 (float): mass of particle 2 in channel 1
        ma2 (float): mass of particle 1 in channel 2
        mb2 (float): mass of particle 2 in channel 2

    Returns:
        complex: Flatté amplitude
    """
    mab = atfi.sqrt(s)
    pab1 = atfk.two_body_momentum(mab, ma1, mb1)
    rho1 = 2.0 * pab1 / mab
    pab2 = atfk.complex_two_body_momentum(mab, ma2, mb2)
    rho2 = 2.0 * pab2 / atfi.cast_complex(mab)
    gamma = (
        atfi.cast_complex(g1 ** 2 * rho1) + atfi.cast_complex(g2 ** 2) * rho2
    ) / atfi.cast_complex(m)
    return relativistic_breit_wigner(s, m, gamma)


@atfi.function
def special_flatte_lineshape(
    m2, m0, gamma0, ma, mb, mc, md, dr, dd, lr, ld, barrier_factor=True
):
    r"""Flatté amplitude with Blatt-Weisskopf formfactors, 2 component mass-dependent width and orbital barriers as done in Pentaquark analysis for L(1405) that peaks below pK threshold.

    .. math::

        SF(m^2) = \frac{1}{m_{res}^2 - m^2 - i m_{res} \Gamma} BWFF(q) BWFF(p_{1}) ( B_r B_d )_{optional}

    
    where

    .. math::

        \Gamma = \Gamma_1 + \Gamma_2

    
    with :math:`\Gamma_1` and :math:`\Gamma_2` are mass-dependent widths for the two channels :math:`R \to a_1 b_1` and :math:`R \to a_2 b_2`, respectively.


    NB: 
    - ma = [ma1, ma2] and mb = [mb1, mb2]
    - The dominant decay for a given resonance should be the 2nd channel i.e. R -> a2 b2. This is because (as done in pentaquark analysis) in calculating :math:`p_0` (used in Blatt-Weisskopf FF) for both channels, the dominant decay is used.
    Another assumption made in pentaquark is equal couplings ie. :math:`\Gamma_{0-1} = \Gamma_{0-2} = \Gamma` and only differ in phase space factors

    Args:
        m2 (float): invariant mass squared of the system
        m0 (float): resonance mass
        gamma0 (float): resonance width
        ma (list): array of masses of particle *a* (2 elements)
        mb (list): array of masses of particle *b* (2 elements)
        mc (float): mass of the other particle (particle *c*)
        md (float): mass of the decaying particle
        dr (float): barrier radius for the resonance
        dd (float): barrier radius for the decay
        lr (int): orbital angular momentum of the resonance
        ld (int): orbital angular momentum of the decay
        barrier_factor (bool, optional): multiplies the form factor for the barrier factors. Defaults to True.

    Returns:
        complex: special Flatté amplitude
    """
    ma1, ma2 = ma[0], ma[1]
    mb1, mb2 = mb[0], mb[1]
    m = atfi.sqrt(m2)
    # D->R c
    q = atfk.two_body_momentum(md, m, mc)
    q0 = atfk.two_body_momentum(md, m0, mc)
    ffd = blatt_weisskopf_ff(q, q0, dd, ld)
    # R -> a1 b1
    p_1 = atfk.two_body_momentum(m, ma1, mb1)
    p0_1 = atfk.two_body_momentum(m0, ma1, mb1)
    ffr_1 = blatt_weisskopf_ff(p_1, p0_1, dr, lr)
    # R -> a2 b2
    p_2 = atfk.two_body_momentum(m, ma2, mb2)
    p0_2 = atfk.two_body_momentum(m0, ma2, mb2)
    ffr_2 = blatt_weisskopf_ff(p_2, p0_2, dr, lr)
    # lineshape
    width_1 = mass_dependent_width(
        m, m0, gamma0, p_1, p0_2, blatt_weisskopf_ff(p_1, p0_2, dr, lr), lr
    )
    width_2 = mass_dependent_width(m, m0, gamma0, p_2, p0_2, ffr_2, lr)
    width = width_1 + width_2
    bw = relativistic_breit_wigner(m2, m0, width)
    # Form factor def
    ff = ffr_1 * ffd
    if barrier_factor:
        b1 = orbital_barrier_factor(p_1, p0_1, lr)
        b2 = orbital_barrier_factor(q, q0, ld)
        ff *= b1 * b2
    return bw * atfi.complex(ff, atfi.const(0.0))


@atfi.function
def nonresonant_lass_lineshape(m2ab, a, r, ma, mb):
    r"""LASS line shape, nonresonant part

    .. math::

        LASS(m^2) = \frac{m}{q \cot \delta_b - i q}

    
    with :math:`q` is the momentum of the two-body system and :math:`\delta_b` is the scattering phase shift

    .. math::

        \cot \delta_b = \frac{1}{a q} + \frac{1}{2} r q


    from `Aston et al. Nuclear Physics B, Volume 296, Issue 3 (1988), Pages 493-526 <https://doi.org/10.1016/0550-3213(88)90028-4>`_

    Args:
        m2ab (float): invariant mass squared of the system
        a (float): parameter of the effective range term
        r (float): parameter of the effective range term
        ma (float): mass of particle a
        mb (float): mass of particle b

    Returns:
        complex: the nonresonant LASS amplitude
    """
    m = atfi.sqrt(m2ab)
    #q = atfk.two_body_momentum(md, m, mc)
    q = atfk.two_body_momentum(m, ma, mb)
    cot_deltab = 1.0 / a / q + 1.0 / 2.0 * r * q
    ampl = atfi.cast_complex(m) / atfi.complex(q * cot_deltab, -q)
    return ampl


@atfi.function
def resonant_lass_lineshape(m2ab,
                            m0,
                            gamma0,
                            a,
                            r,
                            ma,
                            mb,
                            use_mass_dependent_width=False):
    r"""LASS line shape, resonant part

    .. math::

        LASS(m^2) = BW(m^2) (\cos \delta_b + i \sin \delta_b ) ( m^2 \Gamma_0 / q_0 )

    Args:
        m2ab (float): invariant mass squared of the system
        m0 (float): resonance mass
        gamma0 (float): resonance width
        a (float): parameter *a* of the effective range term
        r (float): parameter *r* of the effective range term
        ma (float): mass of particle a
        mb (float): mass of particle b
        mass_dependent_width (bool, optional): if True, the width is mass dependent. Defaults to False.

    Returns:
        complex: the resonant LASS amplitude
    """
    m = atfi.sqrt(m2ab)
    # q = atfk.two_body_momentum(md, m, mc)
    # q0 = atfk.two_body_momentum(md, m0, mc)
    q0 = atfk.two_body_momentum(m0, ma, mb)
    q = atfk.two_body_momentum(m, ma, mb)
    cot_deltab = 1.0 / a / q + 1.0 / 2.0 * r * q
    phase = atfi.atan(1.0 / cot_deltab)
    width = gamma0 * q / m * m0 / q0
    if use_mass_dependent_width:
        width = mass_dependent_width(m, m0, gamma0, q, q0, 1.0, 0)
    ampl = (relativistic_breit_wigner(m2ab, m0, width) *
            atfi.complex(atfi.cos(phase), atfi.sin(phase)) *
            atfi.cast_complex(m2ab * gamma0 / q0))
    return ampl


@atfi.function
def dabba_lineshape(m2ab, b, alpha, beta, ma, mb):
    r"""Dabba line shape

    .. math::

        Dabba(m^2) = \frac{1 - \beta (m^2 - m_{sum}^2) + i b e^{-\alpha (m^2 - m_{sum}^2)}(m^2 - s_{Adler}) \rho}{(1 - \beta (m^2 - m_{sum}^2))^2 + (b e^{-\alpha (m^2 - m_{sum}^2)} (m^2 - s_{Adler}) \rho)^2}


    Args:
        m2ab (float): invariant mass squared of the system
        b (float): parameter b
        alpha (float): parameter alpha
        beta (float): parameter beta
        ma (float): mass of particle a
        mb (float): mass of particle b

    Returns:
        complex: the Dabba amplitude
    """
    mSum = ma + mb
    m2a = ma ** 2
    m2b = mb ** 2
    sAdler = max(m2a, m2b) - 0.5 * min(m2a, m2b)
    mSum2 = mSum * mSum
    mDiff = m2ab - mSum2
    rho = atfi.sqrt(1.0 - mSum2 / m2ab)
    realPart = 1.0 - beta * mDiff
    imagPart = b * atfi.exp(-alpha * mDiff) * (m2ab - sAdler) * rho
    denomFactor = realPart * realPart + imagPart * imagPart
    ampl = atfi.complex(realPart, imagPart) / atfi.cast_complex(denomFactor)
    return ampl


def build_spline_lineshape(degree, nodes):
    """
    Spline interpolated line shape
    """
    from sympy import interpolating_spline, symarray
    from sympy.abc import x
    from sympy.utilities.lambdify import lambdify, lambdastr

    coeffs = symarray("c", len(nodes))
    spl = interpolating_spline(degree, x, nodes, coeffs)
    f = lambdify([x, coeffs], spl, "tensorflow")

    @atfi.function
    def spline_lineshape(m, ampl_re, ampl_im):
        return atfi.complex(f(m, ampl_re), f(m, ampl_im))

    return spline_lineshape
