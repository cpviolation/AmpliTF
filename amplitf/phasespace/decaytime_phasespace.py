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

import math
import numpy as np
import tensorflow as tf
import amplitf.interface as atfi


class DecayTimePhaseSpace:
    """
    Class for decay time phase space.

    Samples are drawn from the exponential decay distribution exp(-t/tau)/tau,
    not uniformly: a PDF used with tfa.toymc.run_toymc or as the integrand of a
    normalisation integral over these samples must be divided by
    :meth:`density` (for mixing, use the amplitf.mixing time-evolution functions
    with ``relative_to_decay=True``).
    """

    def __init__(self, tau=1.0):
        """
        Constructor
        """
        self.tau = tau

    @atfi.function
    def inside(self, x):
        """
        Check if the point x is inside the phase space (x > 0)
        """
        t = self.t(x)
        inside = tf.greater(t, 0)
        return inside

    @atfi.function
    def filter(self, x):
        return tf.boolean_mask(x, self.inside(x))

    #@atfi.function
    def unfiltered_sample(self, size, maximum=None):
        """
        Return a sample of decay times distributed as exp(-t/tau)/tau (see :meth:`density`).
          size     : number of points to generate
          maximum  : if not None, add a 2nd dimension with a uniform number from 0 to maximum.
                     Useful for accept-reject toy MC.
        """
        u = tf.random.uniform([size], 0, 1, dtype=atfi.fptype())
        v = [-atfi.log(1-u) * self.tau]
        if maximum is not None:
            v += [tf.random.uniform([size], 0.0, maximum, dtype=atfi.fptype())]
        return tf.stack(v, axis=1)

    #@atfi.function
    def uniform_sample(self, size, maximum=None):
        """
        Sample of decay times distributed as exp(-t/tau)/tau, i.e. NOT uniform (the name is kept
        for compatibility with the other phase-space classes). See :meth:`density`.
          size     : number of points to generate
          maximum  : if not None, add a 2nd dimension with a uniform number from 0 to maximum.
        """
        return self.filter(self.unfiltered_sample(size, maximum))

    @atfi.function
    def density(self, x):
        """
        Probability density exp(-t/tau)/tau of the samples from :meth:`uniform_sample` and
        :meth:`unfiltered_sample`, at the points x.
        """
        t = self.t(x)
        return atfi.exp(-t / self.tau) / self.tau
    
    @atfi.function
    def unfiltered_acceptance_sample(self, function, pars, size, maximum=None):
        """
        Generate a sample of point within phase space with an acceptance function.
          function : the function to generate the sample with, it should be a TF function
          pars     : parameters for the function
          size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                     so the number of points in the output will be <size.
          maximum  : if maximum>0, add 3rd dimension to the generated tensor which is
                     uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        v = [function([size], *pars, dtype=atfi.fptype())]
        if maximum is not None:
            v += [tf.random.uniform([size], 0.0, maximum, dtype=atfi.fptype())]
        return tf.stack(v, axis=1)
    
    @atfi.function
    def acceptance_sample(self, function, pars, size, maximum=None):
        """
        Generate uniform sample of point within phase space.
          size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                     so the number of points in the output will be <size.
          majorant : if majorant>0, add 3rd dimension to the generated tensor which is
                     uniform number from 0 to majorant. Useful for accept-reject toy MC.
        Note it does not actually generate the sample, but returns the data flow graph for generation,
        which has to be run within TF session.
        """
        return self.filter(self.unfiltered_acceptance_sample(function, pars, size, maximum))

    def dimensionality(self):
        return 1

    def bounds(self):
        return list(self.ranges)
    
    @atfi.function
    def t(self, sample):
        return sample[..., 0]
