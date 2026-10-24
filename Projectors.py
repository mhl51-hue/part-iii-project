'''
Ground state projectors 
https://iopscience.iop.org/article/10.1088/2058-9565/ae2887/pdf
1. wall-Chebyshev projector
2. Eigenstate filter (delta function)
3. Step function
'''
import numpy as np
from numpy.polynomial import chebyshev as Cheb
'''
Cheb.chebval(x, c[, tensor]) will evaluate a chebyshev series at point x
Cheb.chebinterpolate(func, deg[, args]) will interpolate a function at the chebyshev points of the first kind
'''
import matplotlib.pyplot as plt
from scipy.special import erf

'''
1. wall-Chebyshev projector
'''

def wall_cheb_coefs(m):
    '''
    G_m(x) = 1/(1+2m) sum_k (2-d_k0) (-1)^k T_k(x)
    Eq. 13 from paper
    m is the order of expanion
    k runs from 0 to m
    '''
    k = np.arange(m+1)
    coefs = ((2 - (k == 0)) * (-1) ** k) / (1 + 2*m)
    return coefs

def wall_cheb(HEvals, E, R, m):
    '''
    map the Hamiltonian eigenspectrum onto the [-1, 1] interval
    x = 2((H-E)/R) - 1 Eq. 14 from paper
    Use to evaulate G_m

    HEvals is the hamiltonian eigenvalues
    E is the estimate of ground state energy
    R is the spectral range of the Hamiltonian (energy difference between lowest and highest possible energies in system)
    '''
    x = 2 * ((np.asarray(HEvals) - E) / R) - 1
    return Cheb.chebval(x, wall_cheb_coefs(m))

def wall_cheb_nodes(E, R, m):
    '''
    nodes from Eq. 16
    nu is the index integer running from 0 to m used to calculate specific chebyshev nodes
    '''
    nu = np.arange(1, m+1)
    return E + 0.5 * R * (1 - np.cos(nu * np.pi / (m+0.5) )) 


'''
2. Eigenstate filter (delta function)
'''

def delta_func(x, l, Edelta):
    '''
    x is the rescaled energy (E- lambda)/(alpha + |lambda|)
    l is the degree of the chebyshev filter polynomial
    Edelta is the spectral gap of the system (energy difference between ground and first excited state)
    Eq. 9
    '''
    a = np.zeros(l + 1); a[l] = 1 # Array for chebval coefs
    b = lambda z: -1 + 2 * (z**2 - Edelta**2) / (1 - Edelta**2) # Coordinate transformation, maps [Edelta, 1] onto [-1, 1]
    return Cheb.chebval(b(np.asarray(x)), a) / Cheb.chebval(b(0), a)

'''
3. Step function
'''

def step_func(deg, delta, mu, epsilon = 1e-2):
    '''
    deg is the degree of interpolation (number of chebyshev points used to fit the step function)
    delta is the transition width of the step function
    epsilon is the target error tolerance for the step function approximation
    k is a scaling factor derived from delta and epsilon, controls steepness of erf
    '''
    k = np.sqrt(2) / delta * np.sqrt(np.log(2 / (np.pi * epsilon**2)))
    f = lambda x: 0.5 * (1 - erf(k * (x - mu)))
    return Cheb.chebinterpolate(f, deg), k

def step(x, coefs):
    return Cheb.chebval(np.asarray(x), coefs)

'''
plotting the three proctors on their own to check their shapes
'''

x = np.linspace(-1, 1, 2000)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))

# Wall-Chebyshev
for m in (1, 3, 20):
    ax[0].plot(x, Cheb.chebval(x, wall_cheb_coefs(m)), label=f'm={m}')
ax[0].set_title("wall-Chebyshev")

# Eigenstate filter
for l in (5, 20):
    ax[1].plot(x, delta_func(x, l, 0.2), label=f'l={l}')
ax[1].set_title("eigenstate filter, Delta=0.2")

# Step function
for d in (50, 150):
    coefs_step, k = step_func(deg=d, delta=0.075, mu=0)
    ax[2].plot(x, step(x, coefs_step), label=f'degree={d}')
ax[2].set_title("step, delta=0.075")

for a in ax:
    a.set_xlabel("x")
    a.legend()
plt.tight_layout()

if __name__ == "__main__":
    plt.show()