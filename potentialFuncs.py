import numpy as np 


def harmonicOscillator(x:float, y:float, m:float, omega0:float):
    return 1/2 *m*omega0**2 * (x**2+y**2) 

def coloumb(x:float, y:float, m:float, Q:float):
    return Q**2/np.sqrt(x**2+y**2)

def periodic(x:float,y:float,m:float,omega0:float):
    return -6*np.abs(1/(np.cos(omega0*x)*np.cos(omega0*y)))+8 #recommended omega0=2

def uniformGravity(x:float, y:float, m:float, g:float):
    return m*g*y

def effectivePotential(x:float, y:float, m:float, omega0:float):
    return 1/(x+y)**2-omega0/(x+y)

def circularWell(x:float, y:float, E0:float, R:float):
    return np.where(np.sqrt(x**2+y**2)<R, 0, E0)
    
def squareWell(x:float, y:float, E0:float, length:float):
    return np.where(np.logical_and(np.abs(x)<length, np.abs(y)<length), 0, E0)





