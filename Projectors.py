'''
Ground state projectors 
https://iopscience.iop.org/article/10.1088/2058-9565/ae2887/pdf
1. wall-Chebyshev projector
2. Eigenstate filter (delta function)
3. Step function
'''
import numpy as np
from numpy.polynomial import chebyshev as C
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

'''
recreating figure 1

def plot_wall_cheb():
    x = np.linspace(-1.1, 1.1, 1000)
    orders = [1, 2, 3, 4, 5, 10]

    plt.axhline(0, color = 'black', linewidth = 1)
    plt.axvline(1, color = 'black', linewidth = 1)
    plt.axvline(-1, color = 'black', linewidth = 1)
    
    for m in orders:
        y = C.chebval(x, wall_cheb_coefs(m))
        plt.plot(x, y, label=f'$m = {m}$')

    plt.xlim(-1.1, 1.1)
    plt.ylim(-0.6, 2.1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$G_m^{\mathrm{wall-Ch}}(x)$')
    plt.legend()
    plt.tight_layout()

    plt.show()
    
plot_wall_cheb()
'''

def wall_cheb(H, E, R, m):
    '''
    map the Hamiltonian eigenspectrum onto the [-1, 1] interval
    x = 2((H-E)/R) - 1 Eq. 14 from paper
    Use to evaulate G_m

    H is the hamiltonian eigenvalues
    E is the estimate of ground state energy
    R is the spectral range of the Hamiltonian
    '''
    x = 2 * ((np.asarray(H) - E) / R) - 1
    return C.chebval(x, wall_cheb_coefs(m))

def wall_cheb_nodes(E, R, m):
    '''
    nodes from Eq. 16
    '''
    nu = np.arange(1, m+1)
    return E + 0.5 * R * (1 - np.cos(nu * np.pi / (m+0.5) )) 


'''
2. Eigenstate filter (delta function)
'''

def detla_func(x, l, Delta):
    '''
    x is the rescaled energy (E- lambda)/(alpha + |lambda|)
    Delta is the rescaled gap
    Eq. 9
    '''
    a = np.zeros(l + 1); a[l] = 1 #array for chebval coefs
    b = lambda z: -1 + 2 * (z**2 - Delta**2) / (1 - Delta**2)
    return C.chebval(b(np.asarray(x)), a) / C.chebval(b(0), a)

'''
3. Step function
'''

def step_func(deg, delta, eps = 1e-2):

    k = np.sqrt(2) / delta * np.sqrt(np.log(2 / (np.pi * eps**2)))
    f = lambda x: 0.5 * (1 - erf(k * x))
    return C.chebinterpolate(f, deg)

def step(x, coefs):
    return C.chebval(np.asarray(x), coefs)

