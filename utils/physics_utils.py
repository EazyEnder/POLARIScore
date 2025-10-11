import numpy as np

LIGHT_SPEED = 299792458
"""Velocity of the light in m/s"""
PLANCK_CONSTANT = 6.62607e-34
"""Planck constant in J s"""
BOLTZMANN_CONSTANT = 1.380649e-23
"""Boltzmann constant in J/K"""
PC_TO_CM = 3.086e+18
"""How many centimeters in a parsec"""

BLACKBODY_EMISSION = lambda nu,T: (2*PLANCK_CONSTANT*np.power(nu,3)/(LIGHT_SPEED**2))*(1/(np.exp(PLANCK_CONSTANT*nu/(BOLTZMANN_CONSTANT*T))-1))
"""Emmision of a blackbody in function of the frequency and temperature"""
CMB_TEMPERATURE = 2.725

CO_ABUNDANCE = 1e-4
CO_A = [7.203e-8,6.9e-7,2.5e-6]
"""CO line spontaneous emission coefficient in s^-1"""
CO_FREQUENCY= [115.271e9,230.538e9,345.796e9]
"""CO line frequency in Hz"""

CO_ROT_CST = 57.64e9
ROT_ENERGY = lambda l,rot_cst: PLANCK_CONSTANT*rot_cst*l*(l+1)/BOLTZMANN_CONSTANT

GAUSSIAN = lambda x,sigma: (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-np.power(x,2)/(2*sigma**2))

CONVERT_INTENSITY_TO_KELVIN = lambda I,nu: I*LIGHT_SPEED**2 / (2.*BOLTZMANN_CONSTANT*nu**2)

CONVERT_NH_TO_EXTINCTION = lambda c: c/(2*0.94e21) #(Bohlin et al. 1978)
CONVERT_EXTINCTION_TO_NH = lambda a: a*2*0.94e21
