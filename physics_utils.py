import numpy as np

LIGHT_SPEED = 299792458
"""Velocity of the light in m/s"""
PLANCK_CONSTANT = 6.62607e-34
"""Planck constant in J s"""
BOLTZMANN_CONSTANT = 1.380649e-23
"""Boltzmann constant in J/K"""

BLACKBODY_EMISSION = lambda nu,T: (2*PLANCK_CONSTANT*np.power(nu,3)/(LIGHT_SPEED**2))*(1/(np.exp(PLANCK_CONSTANT*nu/(BOLTZMANN_CONSTANT*T))-1))
"""Emmision of a blackbody in function of the frequency and temperature"""

CO_ABUNDANCE = 1e-4
CO_J10_FREQUENCY = 115.271e9
"""CO J=1-0 line frequency in Hz"""
CO_J10_A = 7.203e-8
"""CO J=1-0 line spontaneous emission coefficient in s^-1"""
CO_J10_TEMP = 5.53

GAUSSIAN = lambda x,sigma: (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-np.power(x,2)/(2*sigma**2))