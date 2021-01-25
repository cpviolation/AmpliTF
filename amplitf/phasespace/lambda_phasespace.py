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

import math
import numpy as np
import tensorflow as tf
import amplitf.interface as atfi


class LambdaPhaseSpace:
    """
    Accept/Veto an area over the "origin" phase space based on a boolean lambda function
    """

    def __init__(self, phsp, func):
        self.phsp = phsp
        self.func = func

    def dimensionality(self):
        return self.phsp.dimensionality()

    def inside(self, x):
        return tf.logical_and(self.phsp.inside(x), self.func(x))

    def filter(self, x):
        #        return tf.boolean_mask(x, self.inside(x))
        y = tf.boolean_mask(x, self.func(x))
        return tf.boolean_mask(y, self.phsp.inside(y))

    def unfiltered_sample(self, size, maximum=None):
        """
        Generate uniform sample of point within phase space.
          size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                     so the number of points in the output will be <size.
          maximum  : if majorant>0, add 3rd dimension to the generated tensor which is
                     uniform number from 0 to majorant. Useful for accept-reject toy MC.
        """
        return self.phsp.unfiltered_sample(size, maximum)

    def uniform_sample(self, size, maximum=None):
        """
        Generate uniform sample of point within phase space.
          size     : number of _initial_ points to generate. Not all of them will fall into phase space,
                     so the number of points in the output will be <size.
          maximum  : if majorant>0, add 3rd dimension to the generated tensor which is
                     uniform number from 0 to majorant. Useful for accept-reject toy MC.
        Note it does not actually generate the sample, but returns the data flow graph for generation,
        which has to be run within TF session.
        """
        return self.filter(self.unfiltered_sample(size, maximum))

    def bounds(self):
        return self.phsp.bounds()
