import numpy as np
import vegas

# Constants
m_pi0 = 0.134977 #GeV
f_pi = 0.092 #GeV
F_pi = (4*np.pi*f_pi)**(-1)
alpha = 1./137.
hbar = 6.582119569e-16 * 1e-9  # GeV s
tau_pi0 = 8.52e-17 #s
pi0_width = hbar / tau_pi0 # GeV

def E_N_rest_min(m_N):
    return 1.01*m_N

def E_N_rest_max(m_N):
    return (m_N**2 + m_pi0**2) / (2*m_pi0)

def E_N_rest_bounds(m_N,E_lab,gamma):
  beta = np.sqrt(1 - gamma**(-2))
  P_lab = np.sqrt(E_lab**2 - m_N**2)
  E_rest_minus = gamma * E_lab - beta * gamma * P_lab
  E_rest_plus = gamma * E_lab + beta * gamma * P_lab
  E_a = np.where(E_rest_minus < E_N_rest_min(m_N), E_N_rest_min(m_N), E_rest_minus)
  E_b = np.where(E_rest_plus > E_N_rest_max(m_N), E_N_rest_max(m_N), E_rest_plus)
  return E_a,E_b

def dGamma_dE_rest(E_rest,d,m_N):
  P_rest = np.sqrt(E_rest**2 - m_N**2)
  prefactor = -1./(2*np.pi*m_pi0) * alpha**2 * d**2 * F_pi**2
  term1 = P_rest*(4 * E_rest**2 * m_pi0**2 - 3 * E_rest * m_pi0 * m_N**2 -
             2 * E_rest * m_pi0**3 + m_N**4 + 3 * m_pi0**2 * m_N**2)
  term2 = m_pi0**2 * (4*E_rest - m_pi0) * m_N**2 * np.arctanh(P_rest/E_rest)
  return prefactor * (term1 - term2)

def dGamma_dE_lab(d,m_N,E_lab,gamma):
  beta = np.sqrt(1 - gamma**(-2))
  P_lab = np.sqrt(E_lab**2 - m_N**2)
  E_a,E_b = E_N_rest_bounds(m_N,E_lab,gamma)
  def integrand(E_rest):
    return 1 / (2 * gamma * beta * P_lab) * dGamma_dE_rest(E_rest,d,m_N)
  integ = vegas.Integrator([[E_a,E_b]])
  result = integ(integrand, nitn=10, neval=20)
  return result[0].mean

