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
