# tools for plotting roots of random polynomials
import numpy as np
import scipy.stats
from scipy.special import gamma
import matplotlib.pyplot as plt


def random_coeffs(params,imag=False,add_one=True):
    """
    INPUT
        params (list) tuples specifying distribution type (str), args (list), and kwargs (dict).
    
    RETURNS
        coeffs
    
    """
    dist = {
        'normal':np.random.normal,
        'beta':np.random.beta,
        'uniform':np.random.uniform,
        'gamma':np.random.gamma,
        'exponential':np.random.exponential,
        'lognormal':np.random.lognormal,
        'zeros':np.zeros
    }
    
    coeffs = np.array([])
    for p in params:
        dist_name,params,kwargs = p
        coeffs = np.append(coeffs,dist[dist_name](*params,**kwargs))
        if imag:
            coeffs.astype(np.complex64)
            coeffs.imag = dist[dist_name](*params,**kwargs)
    if add_one:
        coeffs = np.append(coeffs,np.array([1.]))
    return coeffs


def p_type(coeffs, basis='power'):
    """
    Return a polynomial object using the specified polynomial type.
    
    INPUT
        coeffs (array)
        basis (str) the key to specify the polynomial object type
    
    """
    P = {'power':np.polynomial.polynomial.Polynomial,
         'chebyshev':np.polynomial.chebyshev.Chebyshev,
         'power_imag':np.polynomial.polynomial.polyfromroots}
    return P[basis](coeffs)


# generate several random polynomials with various distributions and plot their roots
def poly_roots(params,basis='power',dx=None,plot_range=1,correction=False,return_values=False,niters=1):
    """
    Plot random polynomials and their roots with coefficients specified by params.
    
    INPUTS
        params (list) input to the random_coeffs function
        basis (str) which polynomial type to use
        niters (int) number of polynomials to simulate
    
    """

    colors = 'rgb'
    fig = plt.figure(figsize=(12,6))
    x = np.linspace(-1,1,100) # used for plotting P(x)
    
    for step in range(niters):
        X = [] # to store the real part of the roots
        Y = [] # to store the imaginary part of the roots
        dX = [] # to store the roots of the specified derivative
        dY = [] # store the imaginary part

        plt.subplot(1,2,1) # the first subplot shows the polynomial on the xy plane
        plt.plot(x,np.zeros_like(x),color='grey') # plots the line y=0

        # define the polynomial P
        coeffs = random_coeffs(params)
        P = p_type(coeffs,basis)
        R = P.roots() # get the roots of P(x)
        X.append(R.real)
        Y.append(R.imag)

        if dx:
            for i in range(plot_range,abs(dx)+1):
                # compute the derivative
                if dx >= 1:
                    dP = P.deriv(m=i)
                    #plt.plot(x,dP(x),color='r',alpha=.5)
                    dR = dP.roots()
                    dX.append(dR.real)
                    dY.append(dR.imag)
                # compute the antiderivative and add a standard normal constant
                elif dx <= -1:
                    constants = np.random.randn(abs(i))
                    if correction:
                        constants *= np.array([gamma(abs(dx)-j) for j in range(i)])
                    Pint = P.integ(m=i,k=constants)
                    dR = Pint.roots()
                    dX.append(dR.real)
                    dY.append(dR.imag)

        # plot the random polynomial
        plt.plot(x,P(x),color='k',alpha=.3,lw=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("P(x), degree %s" % (len(coeffs)-2))

        plt.subplot(1,2,2)
        plt.scatter(X,Y,alpha=.4,s=10)
        if dx is not None:
            for dx_,dy_ in zip(dX,dY):
                plt.scatter(dx_,dy_,alpha=.1,s=50,color=colors[step%3])
        plt.title("The Roots of P(x), degree %s" % (len(coeffs)-2))
        plt.xlabel('real')
        plt.ylabel('imag')
    
    if dx is None:
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
    elif dx >= 1:
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
    elif dx <= -1:
        plt.axis('equal')
    plt.show()
    
    if return_values:
        return X,Y,dX,dY


def plot_poly_roots(polylist,extras=None,x=np.linspace(-1,1,101)):
    """
    plot a polynomial P and its roots
    
    P is a list of polynomials
    """
    x = np.linspace(-1,1,101)

    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.plot(x,np.zeros_like(x),color='grey',alpha=.5)

    for P in polylist:
        plt.plot(x,P(x))

    plt.subplot(1,2,2)
    plt.plot(x,np.zeros_like(x),color='grey',alpha=.5)
    plt.plot(np.zeros_like(x),x,color='grey',alpha=.5)
    for P in polylist:
        plt.subplot(1,2,2)
        r = P.roots()
        
        plt.scatter(r.real,r.imag,s=100,alpha=.2)
        plt.axis('equal')
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        
        if extras:
            plt.subplot(1,2,2)
            for e in extras:
                plt.scatter(e.real,e.imag,color='k',alpha=.8)
    plt.show()

    
    
def find_roots_x(d,stop_deg=2,basis='power',correction=True,perturb=None):
    """
    d is the degree of the original random polynomial
    stop_deg is stop after reaching a certain degree
    
    """
    
    if d == 75:
        params = [('normal',[0,1],{'size':25}), ('gamma',[2,2],{'size':25}), ('beta',[.5,.5],{'size':25})]
        coeffs = random_coeffs(params)
    else:
        coeffs = np.random.randn(d+1)
        coeffs[-1] = 1.

    coeffs[-1] = 1
    P = np.polynomial.polynomial.Polynomial(coeffs)
    if basis == 'chebyshev':
        P = P.convert(kind=np.polynomial.Chebyshev)
    
    constants = coeffs[:-stop_deg-1]
    constants = constants[::-1]
    if correction:
        for i in range(len(constants)):
            constants[i] *= gamma(len(constants)-i)
    dP = P.deriv(m=d-stop_deg)
    if perturb:
        dP.coef += np.random.normal(perturb[0],perturb[1],size=len(dP.coef))

    P_ = dP.integ(m=d-stop_deg,k=constants)
    
    plot_poly_roots([P,P_])
    plt.figure(figsize=(16,4))
    plt.title("Polynomial Coefficients")
    plt.bar(np.arange(d+1),P.coef,alpha=.3)
    plt.bar(np.arange(d+1),P_.coef,alpha=.3)
    plt.show()
    
