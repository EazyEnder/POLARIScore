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

from scipy.stats import lognorm
def plot_lognorm(ax, mean, std, amp=1., x_min=1e-2, x_max=1e3, n_points=100, 
                 color='red', label=None, lw=1, ls="--"):
    x = np.logspace(np.log10(x_min), np.log10(x_max), n_points)
    pdf = lognorm.pdf(x, s=std, scale=mean)
    pdf = pdf * x
    ax.plot(x, amp*pdf, color=color, lw=lw, label=label, linestyle=ls)
    return ax

def dcmf_func(M, amp, mu, sigma, alpha, cutoff):
        pdf_low = lognorm.pdf(M, s=sigma, scale=np.abs(mu))

        pdf_high = M**(-2.3)
        join_mass = cutoff
        scale_factor = (pdf_low[np.argmin(np.abs(M - join_mass))] /
                        pdf_high[np.argmin(np.abs(M - join_mass))])
        pdf_high *= scale_factor
        amp_scaled = amp#/np.max(pdf_low)

        pdf_low *= amp_scaled
        pdf_high *= amp_scaled

        if type(M) is np.ndarray or type(M) is list:
            return np.concatenate((pdf_low[M <= cutoff],pdf_high[M > cutoff]),axis=0)*M
        else:
          if M >  cutoff:
              return pdf_high*M
          else:
              return pdf_low*M

def plot_imf_chabrier(ax, color='black', x_min=1e-2, x_max=1e3, n_points=100,
                      Mc=0.22, sigma_ln=1.31, alpha=2.3, amp=25.0):

    x = np.logspace(np.log10(x_min), np.log10(x_max), n_points)

    pdf_low = lognorm.pdf(x, s=sigma_ln, scale=Mc)
    pdf_low = pdf_low * x * np.log(10)

    pdf_high = x**(1 - alpha)

    join_mass = 1.0
    scale_factor = (pdf_low[np.argmin(np.abs(x - join_mass))] /
                    pdf_high[np.argmin(np.abs(x - join_mass))])
    pdf_high *= scale_factor

    amp_scaled = amp/np.max(pdf_low)

    pdf_low *= amp_scaled
    pdf_high *= amp_scaled

    ax.plot(x[x <= join_mass], pdf_low[x <= join_mass],
            ls='--', color=color, label='IMF (Chabrier, 2005)')
    ax.plot(x[x > join_mass], pdf_high[x > join_mass],
            ls='--', color=color)

    return ax
    
    