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

# CCM pi0 infomration
pi0_momentum_bins = np.linspace(0,0.6,31)
pi0_momentum_dist = np.loadtxt('CCM_pi0_momentum.csv')[:,1] * (pi0_momentum_bins[1]-pi0_momentum_bins[0])

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
  return prefactor * (term1 - term2) / pi0_width

def dGamma_dE_lab(E_lab,d,m_N,gamma):
  beta = np.sqrt(1 - gamma**(-2))
  P_lab = np.sqrt(E_lab**2 - m_N**2)
  E_a,E_b = E_N_rest_bounds(m_N,E_lab,gamma)
  def integrand(E_rest):
    #dGamma = 1 / (2 * gamma * beta * P_lab) * dGamma_dE_rest(E_rest,d,m_N)
    dGamma = 1 / (2 * gamma * beta * np.sqrt(E_rest**2 - m_N**2)) * dGamma_dE_rest(E_rest,d,m_N)
    return max([dGamma,0])
  integ = vegas.Integrator([[E_a,E_b]])
  result = integ(integrand, nitn=10, neval=20)
  return result[0].mean

def N_HNL(E_lab,d,m_N):
  N = 0
  for N_p_pi0,p_low,p_high in zip(pi0_momentum_dist,pi0_momentum_bins[:-1],pi0_momentum_bins[1:]):
    gamma_low,gamma_high = np.sqrt(p_low**2/m_N**2 + 1), np.sqrt(p_high**2/m_N**2 + 1)
    def integrand(gamma):
      return dGamma_dE_lab(E_lab,d,m_N,gamma)
    integ = vegas.Integrator([[gamma_low,gamma_high]])
    result = integ(integrand, nitn=10, neval=20)
    N += max(result.mean*N_p_pi0,0)
  return N





