# tools for plotting roots of random polynomials
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def random_coeffs(params,imag=False):
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
def poly_roots(params,basis='power',dx=None,plot_range=1,return_values=False):
    """
    Plot random polynomials and their roots with coefficients specified by params.
    
    INPUTS
        params (list) input to the random_coeffs function
        basis (str) which polynomial type to use
        niters (int) number of polynomials to simulate
    
    """

    
    fig = plt.figure(figsize=(12,6))
    x = np.linspace(-1,1,100) # used for plotting P(x)
    
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
                plt.plot(x,dP(x),color='r',alpha=.5)
                dR = dP.roots()
                dX.append(dR.real)
                dY.append(dR.imag)
            # compute the antiderivative and add a standard normal constant
            elif dx <= -1:
                Pint = P.integ(m=i,k=np.random.randn(abs(i)))
                dR = Pint.roots()
                dX.append(dR.real)
                dY.append(dR.imag)
            
    # plot the random polynomial
    plt.plot(x,P(x),color='k',alpha=.7,lw=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("P(x), degree %s" % (len(coeffs)-1))

    plt.subplot(1,2,2)
    plt.scatter(X,Y,alpha=.8,s=10)
    if dx is not None:
        for dx_,dy_ in zip(dX,dY):
            plt.scatter(dx_,dy_,alpha=.3,s=10,color='r')
    plt.title("The Roots of P(x), degree %s" % (len(coeffs)-1))
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

