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
Formulas for building a model where meson mixing is involved.

Conventions (as in HFLAV): the mass eigenstates of a neutral meson are

.. math::

    |M_{1,2}\rangle = p|M^0\rangle \pm q|\bar{M}^0\rangle,

with masses :math:`m_{1,2}` and widths :math:`\Gamma_{1,2}`, and the mixing parameters are

.. math::

    \begin{align}
    x &\equiv \frac{m_1-m_2}{\Gamma}, &
    y &\equiv \frac{\Gamma_1-\Gamma_2}{2\Gamma}, &
    \Gamma &\equiv \frac{\Gamma_1+\Gamma_2}{2} = \frac{1}{\tau}.
    \end{align}

The amplitude of an initial :math:`M^0` decaying at time :math:`t` to a final state :math:`f` is

.. math::

    A(t) = A_1 e^{-i m_1 t - \Gamma_1 t/2} + A_2 e^{-i m_2 t - \Gamma_2 t/2},\qquad
    A_{1,2} = \frac{1}{2}\left(A_f \pm \frac{q}{p}\bar{A}_f\right),

so that

.. math::

    |A(t)|^2 = |A_1|^2\psi_+(t) + |A_2|^2\psi_-(t) + 2\,\mathrm{Re}\left(A_1 A_2^* \psi_i(t)\right),

with :math:`\psi_\pm(t) = e^{-(1\pm y)t/\tau}` and :math:`\psi_i(t) = e^{-(1+ix)t/\tau}`.
For an initial :math:`\bar{M}^0`, exchange :math:`A_f \leftrightarrow \bar{A}_f` and
:math:`q/p \to p/q`.

Sampling the decay time: :class:`amplitf.phasespace.decaytime_phasespace.DecayTimePhaseSpace`
draws times from :math:`e^{-t/\tau}/\tau`. Toy generation and normalisation integrals then need
the density *relative to* that exponential, i.e. the time-evolution functions with
``relative_to_decay=True``:
:math:`\psi_\pm/e^{-t/\tau} = e^{\mp y t/\tau}` and :math:`\psi_i/e^{-t/\tau} = e^{-ixt/\tau}`.
Evaluated on data, the relative density differs from the true one by the factor
:math:`e^{-t/\tau}`, which does not depend on :math:`x`, :math:`y` or :math:`q/p`
(it does depend on :math:`\tau` if that is a free parameter).
"""
import tensorflow as tf
import amplitf.interface as atfi
import amplitf.dynamics as atfd


# Time evolution functions.
@atfi.function
def psip(t, y, tau, relative_to_decay=False):
    r"""Time evolution of the :math:`|A_1|^2` term, :math:`\psi_+(t) = e^{-(1+y)t/\tau}`

    Args:
        t (float): decay time of the candidate
        y (float): mixing parameter :math:`y`
        tau (float): lifetime of the decaying particle
        relative_to_decay (bool): if True, return :math:`\psi_+(t)/e^{-t/\tau} = e^{-yt/\tau}`

    Returns:
        float: the time evolution function
    """
    rate = y if relative_to_decay else 1.0 + y
    return atfi.exp(-rate * t / tau)


@atfi.function
def psim(t, y, tau, relative_to_decay=False):
    r"""Time evolution of the :math:`|A_2|^2` term, :math:`\psi_-(t) = e^{-(1-y)t/\tau}`

    Args:
        t (float): decay time of the candidate
        y (float): mixing parameter :math:`y`
        tau (float): lifetime of the decaying particle
        relative_to_decay (bool): if True, return :math:`\psi_-(t)/e^{-t/\tau} = e^{+yt/\tau}`

    Returns:
        float: the time evolution function
    """
    rate = -y if relative_to_decay else 1.0 - y
    return atfi.exp(-rate * t / tau)


@atfi.function
def psii(t, x, tau, relative_to_decay=False):
    r"""Time evolution of the interference term, :math:`\psi_i(t) = e^{-(1+ix)t/\tau}`

    Args:
        t (float): decay time of the candidate
        x (float): mixing parameter :math:`x`
        tau (float): lifetime of the decaying particle
        relative_to_decay (bool): if True, return :math:`\psi_i(t)/e^{-t/\tau} = e^{-ixt/\tau}`

    Returns:
        complex: the time evolution function
    """
    real = atfi.const(0.0) if relative_to_decay else atfi.const(1.0)
    return atfi.exp(-atfi.complex(real, x) * tf.cast(t / tau, atfi.ctype()))


# Probability density
def mixing_density(ampl_dir, ampl_cnj, qoverp,
                   time_evolution_pos, time_evolution_neg, time_evolution_int):
    r"""Calculates the probability density of the mixing

    ..math::

        |\braket{f|\mathcal{H}|\Mz(t)}|^2=\psi_+(t)|A_1|^2 + \psi_-(t)|A_2|^2 + 2\Real\left(\psi_i(t)A_1A^*_2\right)

    where :math:`\psi_+(t)` and :math:`\psi_-(t)` are the time evolution functions for the direct and conjugate decays, respectively, and :math:`\psi_i(t)` is the interference term.
    The amplitudes :math:`A_1` and :math:`A_2` are the sum and difference of the direct and conjugate decay amplitudes:

    ..math::

        A_1 = \frac{1}{2}\left(\Af + \frac{q}{p}\Abf\right)\qquad A_2 = \frac{1}{2}\left(\Af - \frac{q}{p}\Abf\right)

    Args:
        ampl_dir (complex): amplitude of the direct decay
        ampl_cnj (complex): amplitude of the charged conjugate decay
        qoverp (complex): CP violation
        time_evolution_pos (float): time evolution of the sum of the decays
        time_evolution_neg (float): time evolution of the subtraction of the decays
        time_evolution_int (complex): time evolution of the interference of the decays

    Returns:
        float: the density of probability for the event
    """    
    ampDir = ampl_dir
    ampCnj = qoverp * ampl_cnj

    apb2 = 0.5 * (ampDir + ampCnj)
    amb2 = 0.5 * atfi.conjugate(ampDir - ampCnj)

    dens  = atfd.density( apb2 ) * time_evolution_pos
    dens += atfd.density( amb2 ) * time_evolution_neg
    dens += 2.0 * tf.math.real( apb2 * amb2 * time_evolution_int )
    return dens
