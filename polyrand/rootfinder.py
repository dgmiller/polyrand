import numpy as np
from scipy.special import gamma
from scipy.optimize import newton
from matplotlib import pyplot as plt
from matplotlib import animation,rc
from IPython.display import HTML


def G(k,a):
    return (gamma(k+1)/gamma(k-a+1))

def F(k,a):
    return (gamma(k-a+1)/gamma(k+1))

def frac_deriv(coeffs,a,return_coeffs=False):
    """
    The Matrix Fractional Derivative of a polynomial with given coefficients
    Returns the companion matrix of the fractional derivative of the polynomial with coeffs
    
    """    
    n = int(np.floor(a)) # the integer to differentiate
    k = len(coeffs) - 1 # degree of polynomial
    
    if return_coeffs:
        
        dx_coeffs = coeffs[n:].copy()
        dx_coeffs *= np.array([G(i,a) for i in range(n,k+1)])
        return dx_coeffs
    
    else:
        
        # adjust the leading coefficient to be 1.
        if coeffs[-1] != 1:
            coeffs /= coeffs[-1]
            
        # Build the companion matrix of the full polynomial
        C = np.diag(np.ones(k-1),-1) # companion matrix
        C[:,-1] = -coeffs[:-1]

        # Cut the companion matrix by the integer derivative amount
        D = C[n:,n:]
        D[:,-1] /= G(k,a)
        D[:,-1] *= np.array([G(i,a) for i in range(n,k)])
        return D


def frac_antideriv(coeffs,a,const):
    """
    Compute the fractional antiderivative of a polynomial
    
    """
    n = int(np.ceil(a))
    k = len(coeffs) + len(const) - 1
    
    # modify the coefficients
    coeffs *= np.array([F(i,a) for i in range(n,k)])
    const /= np.array([gamma(i) for i in range(1,len(const)+1)])
    return np.append(const,coeffs)


def frac_deriv_poly(P,a,x,roots=False):
    """
    Take the ath fractional derivative of polynomial P.
    Uses numpy polynomial class.
    
    """
    d_coeffs = np.array([G(i,a) for i in range(P.degree()+1)])
    coeffs = d_coeffs*P.coef
    coeffs[:int(np.floor(a))] = 0
    fdP = np.polynomial.polynomial.Polynomial(coeffs)
    if roots:
        return fdP.roots()
    else:
        dP = lambda x: fdP(x)*(x**-a)
        return dP(x)
    

def plot_frac_deriv_example(n=15):
    """
    Plot an example of a fractional derivative.
    
    """
    coeffs = np.array([0.,0.,1.])
    coeffs[-1] = 1.
    P = np.polynomial.polynomial.Polynomial(coeffs)
    dP = P.deriv(m=1)
    x = np.linspace(.0001,1,101)

    plt.figure(figsize=(8,8))
    plt.plot(x,P(x),color='k')
    plt.plot(x,dP(x),color='r')
    for a in np.linspace(.01,2,n):
        plt.plot(x,frac_deriv_poly(P,a,x),color='red',alpha=.3)
    plt.title("Example of the Fractional Derivatives of $f(x) = x^2$")
    plt.show()
    

def timelapse(coeffs,start,stop,n_frames,basis='power',plot_axis=True):
    
    if basis == 'power':
        P = np.polynomial.polynomial.Polynomial(coeffs)
    elif basis == 'chebyshev':
        P = np.polynomial.chebyshev.Chebyshev(coeffs)
    else:
        raise ValueError("%s is not a polynomial class" % basis)

    plt.figure(figsize=(8,8))
    if plot_axis:
        x = np.linspace(-1,1,101)
        zrs = np.zeros_like(x)
        plt.plot(x,zrs,color='grey',alpha=.1)
        plt.plot(zrs,x,color='grey',alpha=.1)
        plt.scatter([0],[0],color='grey',alpha=.3,s=150)
    
    fd_term = np.linspace(start,stop,n_frames)

    for a in fd_term:
        if a == 0:
            PR = P.roots()
            plt.scatter(PR.real,PR.imag,color='k')
        elif a == fd_term[-1]:
            dPR = P.deriv(m=a).roots()
            plt.scatter(dPR.real,dPR.imag,color='r')
        else:
            D = frac_deriv(P.coef,a)
            ew = np.linalg.eig(D)[0]
            plt.scatter(ew.real,ew.imag,color='orange',alpha=.1)
    plt.title("Degree %s polynomial" % P.degree())
    plt.axis('equal')
    plt.show()
    
    

def animate_roots(coeffs,start=None,stop=None,n_frames=200,t_interval=75,plot_trail=True,plot_range=[(-1.5,1.5),(-1.5,1.5)]):
    """
    Animate the continuous deformation of the roots of fractional derivatives.
    
    """
    if start == None:
        start = 0.
    if stop == None:
        stop = len(coeffs) - 2

    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    
    plt.xlim(plot_range[0])
    plt.ylim(plot_range[1])

    points, = plt.plot([], [], color='r', marker='o', ls='None')

    line, = plt.plot([], [], color='orange', marker='o', ls='None', alpha=.08)
    greyline, = plt.plot([], [], color='grey', marker='o', ls='None', alpha=.02)

    # initialization function: plot the background of each frame
    def init():
        greyline.set_data([],[])
        line.set_data([], [])
        points.set_data([], [])
        return (greyline, line, points,)

    # animation function. This is called sequentially
    def animate(i):
        #f = lambda x: np.log(x+1)/np.log(100)
        a = np.linspace(start,stop,n_frames)
        
        # plot trail
        if plot_trail:
            # store the roots that will make the orange trail
            trail = np.array([])
            # store the roots that will make the grey trail
            greytrail = np.array([])
            
            # make the orange trail
            for j in range(max(i-25,0),i):
                fD_past = frac_deriv(coeffs,a[j])
                f_ew_past = np.linalg.eig(fD_past)[0]
                trail = np.append(trail,f_ew_past)
            line.set_data(trail.real, trail.imag)
            
            # make the grey trail
            for j in range(max(i-25,0)):
                fD_past = frac_deriv(coeffs,a[j])
                f_ew_past = np.linalg.eig(fD_past)[0]
                greytrail = np.append(greytrail,f_ew_past)
            greyline.set_data(greytrail.real, greytrail.imag)
                
        # get the current roots (red)
        fD = frac_deriv(coeffs,a[i])
        f_ew = np.linalg.eig(fD)[0]
        points.set_data(f_ew.real, f_ew.imag)

        
        return (greyline, line, points,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=t_interval, blit=False)

    return HTML(anim.to_html5_video())



def cheb_nodes(a,b,n):
    """
    get the chebyshev nodes between 0 and 1
    
    """
    nodes = []
    for k in range(1,n+1):
        nodes.append(.5*(b+a) + .5*(b-a)*np.cos((2*k-1)*np.pi/(2*n)))
    return np.sort(np.array(nodes))[::-1]



def newton_integration(coeffs,n_steps,dx=3):
    """
    NIRF (Integration)
    NARF (Antiderivative)
    N*RF (Newton * Root Finding)
    Use newton's method to integrate the roots of a polynomial up to a different polynomial.
    
    """
    raise NotImplementedError
