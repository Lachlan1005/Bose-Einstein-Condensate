import numpy as np 

def oneGaussian(x,y,t=0):
    return np.exp(-(x**2+y**2))

def wideGaussian(x,y,t=0):
    r=np.sqrt(x**2+y**2)
    return np.exp(-.1*((r-1/3)**2))

def shiftedGaussian(x,y,t=0):
    return np.exp(-3*((x-1/2)**2+(y-1/2)**2))

def thinGaussian(x,y,t=0):
    return np.exp(-5*(x**2+y**2))

def twoGaussian(x,y,t=0):
    return  np.exp(-3*((x-2/3)**2+(y-2/3)**2))+np.exp(-3*((x+2/3)**2+(y+2/3)**2))

def vortex1(x,y,t=0):
    """
    Return the standard vortex form of
    psi=exp(i l phi) f(r)
    where f~r/sqrt{2+r^2}
    """
    l=1
    return np.exp(1j*l*np.arctan2(y, x))*np.sqrt(x**2+y**2)/np.sqrt(x**2+y**2+2)

def vortex_with_gasussian(x,y,t=0):
    return np.exp(1j*10*np.arctan2(y, x))*np.sqrt(x**2+y**2) + 1.69*np.exp(-5*((x-0)**2+y**2))

def normalised_vortex(x,y,t=0):
    l=1
    return np.exp(-3/2 *np.sqrt(x**2+y**2))*np.exp(1j*l*np.arctan2(y, x))*np.sqrt(x**2+y**2)/np.sqrt(x**2+y**2+2)

def thomas_fermi(x,y,t=0):
    """
    Return a Thomas-Fermi profile 
    """
    R=1
    l=2
    r=np.sqrt(x**2+y**2)
    theta=np.arctan2(y,x)

    interior= r<=R

    psi = np.full_like(r, fill_value=10e-5 + 0j, dtype=np.complex64)
    psi[interior]= np.sqrt(1-(r[interior]/R)**2)

    return psi

def thomas_fermiVortex(x,y,t=0):
    """
    Return a Thomas-Fermi profile with rotating phase
    """
    R=1
    l=2
    r=np.sqrt(x**2+y**2)
    theta=np.arctan2(y,x)

    interior= r<=R

    psi = np.full_like(r, fill_value=10e-5 + 0j, dtype=np.complex64)
    psi[interior]= np.exp(1j*l*theta[interior])*np.sqrt(1-(r[interior]/R)**2)

    return psi


def softened_thomas_fermi(x,y,t=0):
    """
    Return a softened Thomas-Fermi profile (Gaussian approximation) with rotating phase
    """
    r=np.sqrt(x**2+y**2)
    theta=np.arctan2(y,x)
    l=1
    n=2
    return np.exp(1j*l*theta)*np.exp(-n*r**2)

def vortexGaussian(x,y,t=0):
    l=1
    n=2
    r=np.sqrt(x**2+y**2)
    theta=np.arctan2(y,x)
    return np.exp(1j*l*theta)*r*np.exp(-n*r**2)
