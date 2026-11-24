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
Covariant (Lorentz-tensor) spin factors, following the conventions of AmpGen
(``src/Vertex.cpp``, https://github.com/GooFit/AmpGen).

Tensors are stored with upper (contravariant) indices, components ordered
:math:`(x, y, z, t)` as in :mod:`amplitf.kinematics`, with a leading batch
dimension: vectors have shape ``[N, 4]`` and rank-2 tensors ``[N, 4, 4]``.
The metric is :math:`g = \mathrm{diag}(-1, -1, -1, +1)`.

For a two-body vertex :math:`R \to a\,b`, ``P`` is :math:`p_a + p_b` and ``Q`` is
:math:`p_a - p_b`; ``V1`` and ``V2`` are the spin tensors of :math:`a` and
:math:`b` (a ``[N]`` scalar for spin 0). The vertex functions are named as in
AmpGen: ``<R>_<a><b>_<L>`` with S/V/T for spin 0/1/2 and S/P/D for the orbital
angular momentum. The order of the daughters matters (it fixes the sign of
``Q`` and of the P-wave terms) and must follow AmpGen's ordering.

Levi-Civita: AmpGen defines :math:`\epsilon^{xyzt} = +1` and contracts it with
all indices lowered, :math:`\epsilon_{xyzt} = \det g = -1`, which is what
:func:`levi_civita` returns.
"""

import itertools

import numpy as np
import tensorflow as tf

import amplitf.interface as atfi

_METRIC = np.diag([-1.0, -1.0, -1.0, 1.0])


def _levi_civita_lower():
    eps = np.zeros((4, 4, 4, 4))
    for perm in itertools.permutations(range(4)):
        eps[perm] = np.linalg.det(np.eye(4)[list(perm)])
    return -eps  # all four indices lowered with g = diag(-1,-1,-1,+1)


_EPS_LOWER = _levi_civita_lower()


def metric():
    """Metric tensor :math:`g^{\\mu\\nu} = g_{\\mu\\nu}` as a ``[4, 4]`` constant."""
    return tf.constant(_METRIC, dtype=atfi.fptype())


def levi_civita():
    """Levi-Civita tensor with lower indices, :math:`\\epsilon_{xyzt} = -1` (AmpGen convention)."""
    return tf.constant(_EPS_LOWER, dtype=atfi.fptype())


def lower(v):
    """Lower the index of a vector, :math:`v_\\mu = g_{\\mu\\nu} v^\\nu`."""
    return v * tf.constant(np.diag(_METRIC), dtype=v.dtype)


def dot(a, b):
    """Minkowski product :math:`a^\\mu g_{\\mu\\nu} b^\\nu` of two batches of vectors."""
    return tf.reduce_sum(a * lower(b), axis=-1)


def outer(a, b):
    """Rank-2 tensor :math:`a^\\mu b^\\nu`."""
    return tf.einsum("...i,...j->...ij", a, b)


def spin1_projector(p):
    """:math:`S^{\\mu\\nu} = g^{\\mu\\nu} - p^\\mu p^\\nu / p^2`."""
    return metric() - outer(p, p) / dot(p, p)[..., None, None]


def orbital_pwave(p, q):
    """P-wave orbital tensor :math:`L^\\mu = q^\\mu - p^\\mu (p\\cdot q)/p^2`."""
    return q - p * (dot(p, q) / dot(p, p))[..., None]


def orbital_dwave(p, q):
    """D-wave orbital tensor :math:`L^\\mu L^\\nu - (L\\cdot L)/3\\, S^{\\mu\\nu}`."""
    l1 = orbital_pwave(p, q)
    return outer(l1, l1) - (dot(l1, l1) / 3.0)[..., None, None] * spin1_projector(p)


def _contract_lower(t, a, b):
    """:math:`t^{\\mu\\nu} a_\\mu b_\\nu` for upper-index vectors ``a``, ``b``."""
    return tf.einsum("...mn,...m,...n->...", t, lower(a), lower(b))


def _lower2(t):
    """Lower both indices of a rank-2 tensor."""
    g = tf.constant(np.diag(_METRIC), dtype=t.dtype)
    return t * g[:, None] * g[None, :]


# Vertices R -> a b. Names and expressions follow AmpGen's Vertex.cpp.

def S_SS_S(P, Q, V1, V2):
    """Scalar to two scalars: :math:`V_1 V_2`."""
    return V1 * V2


def S_VV_S(P, Q, V1, V2):
    """Scalar to two vectors, S-wave: :math:`V_1 \\cdot V_2`."""
    return dot(V1, V2)


def S_VV_P(P, Q, V1, V2):
    """Scalar to two vectors, P-wave: :math:`\\epsilon_{\\mu\\nu\\alpha\\beta} V_1^\\mu V_2^\\nu L^\\alpha P^\\beta`."""
    L = orbital_pwave(P, Q)
    return tf.einsum("mnab,...m,...n,...a,...b->...", levi_civita(), V1, V2, L, P)


def S_VV_D(P, Q, V1, V2):
    """Scalar to two vectors, D-wave: :math:`V_{1\\mu} L_2^{\\mu\\nu} V_{2\\nu}`."""
    return _contract_lower(orbital_dwave(P, Q), V1, V2)


def S_VS_P(P, Q, V1, V2):
    """Scalar to vector + scalar, P-wave: :math:`(V_1 \\cdot L)\\, V_2`."""
    return dot(V1, orbital_pwave(P, Q)) * V2


def S_TV_P(P, Q, V1, V2):
    """Scalar to tensor + vector, P-wave: :math:`T^{\\mu\\nu} L_\\mu V_{2\\nu}`."""
    return _contract_lower(V1, orbital_pwave(P, Q), V2)


def S_TS_D(P, Q, V1, V2):
    """Scalar to tensor + scalar, D-wave: :math:`L_2^{\\mu\\nu} T_{\\mu\\nu} V_2`."""
    return tf.reduce_sum(orbital_dwave(P, Q) * _lower2(V1), axis=[-2, -1]) * V2


def V_SS_P(P, Q, V1, V2):
    """Vector to two scalars, P-wave: :math:`L^\\mu V_1 V_2`."""
    return orbital_pwave(P, Q) * (V1 * V2)[..., None]


def T_SS_D(P, Q, V1, V2):
    """Tensor to two scalars, D-wave: :math:`L_2^{\\mu\\nu} V_1 V_2`."""
    return orbital_dwave(P, Q) * (V1 * V2)[..., None, None]


def V_VS_S(P, Q, V1, V2):
    """Vector to vector + scalar, S-wave: :math:`S^{\\mu\\nu} V_{1\\nu} V_2`."""
    return tf.einsum("...mn,...n->...m", spin1_projector(P), lower(V1)) * V2[..., None]


def V_VS_D(P, Q, V1, V2):
    """Vector to vector + scalar, D-wave: :math:`S^{\\mu\\nu} L_{2,\\nu\\alpha} V_1^\\alpha V_2`."""
    t = tf.einsum("...mn,...na,...a->...m", spin1_projector(P), _lower2(orbital_dwave(P, Q)), V1)
    return t * V2[..., None]


VERTICES = {
    "S_SS_S": S_SS_S,
    "S_TS_D": S_TS_D,
    "V_VS_D": V_VS_D,
    "S_VV_S": S_VV_S,
    "S_VV_P": S_VV_P,
    "S_VV_D": S_VV_D,
    "S_VS_P": S_VS_P,
    "S_TV_P": S_TV_P,
    "V_SS_P": V_SS_P,
    "T_SS_D": T_SS_D,
    "V_VS_S": V_VS_S,
}


def vertex(name, P, Q, V1, V2):
    """Evaluate the vertex ``name`` (e.g. ``"S_VV_P"``)."""
    if name not in VERTICES:
        raise NotImplementedError(f"Vertex {name} is not implemented (available: {sorted(VERTICES)})")
    return VERTICES[name](P, Q, V1, V2)
