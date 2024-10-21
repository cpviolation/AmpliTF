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

"""
Formulas for building a model where meson mixing is involved.

The mixing parameters for the mass eigenstates of a neutral meson ($M^0, \bar{M}^0$) are

$$
\begin{align}
x &\equiv 2\frac{m_2-m_1}{\gamma_1+\gamma_2} = \frac{\Delta M}{\Gamma}\\
y &\equiv \frac{\gamma_2-\gamma_1}{\gamma_1+\gamma_2} = \frac{\Delta\Gamma}{2\Gamma},
\end{align}
$$

where $m_i$ and $\gamma_i$ ($i=1,2$) are the eigenvalues of the phyical eigenstates $M_{1,2}$.

The time evolution of the physical states is given by

$$
|\M_i(t)>= e^{-i(m_i-i\gamma_i/2)t}|\M_i(t=0)>.
$$

... Continue explanation later
"""
import amplitf.interface as atfi

class Mixing:

    def __init__(
            self,
            ampDir : complex,
            ampCnj : complex,
            tau : float,
            x=0.,
            y=0.,
            cpv=0.,
            cpv_phase=1.
    ):
        """Constructor

        Args:
            ampDir (complex): the amplitude of the direct decay model
            ampCnj (complex): the amplitude of the conjugate decay model
            tau (float): the decay time of the neutral meson.
            x (float, optional): the normalised mass difference. Defaults to 0.
            y (float, optional): the normalised width difference. Defaults to 0.
            cpv (float, optional): the amplitude of CP violation (|q/p|). Defaults to 0.
            cpv_phase (float, optional): the pahse of CP violation (arg(q\p)). Defaults to 1.
        """
        self.ampDir = ampDir
        self.ampCnj = ampCnj
        self.tau = tau
        self.x = x
        self.y = y
        self.cpv = cpv
        self.cpv_phase = cpv_phase
        return


    def gamma(self):
        """The inverse of the decay time of the neutral meson

        Returns:
            float: the inverse of the decay time
        """        
        return 1. / self.tau


    # Time evolution functions.
    def psip( 
            self, 
            t : float 
            ):
        """Time evolution function $\psi_+(t)$

        Args:
            t (float): decay time of the candidate

        Returns:
            float: the time evolution function for the sum of the two decay amplitudes
        """
        return atfi.exp( - ( 1.0 - self.x ) * self.gamma() * t )


    def psim( 
            self, 
            t : float 
            ):
        """Time evolution function $\psi_-(t)$

        Args:
            t (float): decay time of the candidate

        Returns:
            float: the time evolution function for the difference of the two decay amplitudes
        """
        return atfi.exp( - ( 1.0 + self.x ) * self.gamma() * t )


    def psii( 
            self, 
            t : float 
            ):
        """Time evolution function $\psi_i(t)$

        Args:
            t (float): decay time of the candidate

        Returns:
            float: the time evolution function for the interference of the two decay amplitudes
        """
        return atfi.exp( - atfi.complex( 1.0, - self.y ) * self.gamma() * t )


    def amplitude(
            self,
            t
            ):
        qoverp = atfi.complex( self.cpv * atfi.cos(self.cpv_phase), 
                               self.cpv * atfi.sin(self.cpv_phase) )
        ampDir = self.ampDir
        ampCnj = qoverp * self.ampCnj

        apb2 = 0.5 * (ampDir + ampCnj)
        amb2 = 0.5 * atfi.conjugate(ampDir - ampCnj)

        ampSq = 0.0
        ampSq += atfi.density( apb2 ) * self.psip( t )
        ampSq += atfi.density( amb2 ) * self.psim( t )
        ampSq += 2.0 * atfi.cast_real( apb2 * amb2 * self.psii( t ) )
        return ampSq