# tools for plotting roots of random polynomials
import numpy as np
import scipy.stats
from scipy.special import gamma
import matplotlib.pyplot as plt
from matplotlib import animation,rc
from IPython.display import HTML


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
def poly_roots(coeffs,basis='power',dx=None,plot_range=1,correction=False,return_values=False,plot_lim=[(-1.5,1.5),(-1.5,1.5)],niters=1):
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
        #coeffs = random_coeffs(params)
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
        plt.title("P(x), degree %s" % (len(coeffs)-1))

        plt.subplot(1,2,2)
        plt.scatter(X,Y,alpha=.4,s=10)
        if dx is not None:
            for dx_,dy_ in zip(dX,dY):
                plt.scatter(dx_,dy_,alpha=.1,s=10,color=colors[step%3],edgecolors=None)
        plt.title("The Roots of P(x), degree %s" % (len(coeffs)-2))
        plt.xlabel('real')
        plt.ylabel('imag')
    
    if dx is None:
        plt.xlim(plot_lim[0])
        plt.ylim(plot_lim[1])
    elif dx >= 1:
        plt.xlim(plot_lim[0])
        plt.ylim(plot_lim[1])
    elif dx <= -1:
        plt.axis('equal')
    plt.show()
    
    if return_values:
        return X,Y,dX,dY

    
# generate several random polynomials with various distributions and plot their roots
def poly_deriv_roots(coeffs,basis='power',dx=1,niters=1,show_dx_roots=True):
    """
    Plot random polynomials and their roots with coefficients specified by params.
    
    INPUTS
        params (list) input to the random_coeffs function
        basis (str) which polynomial type to use
        niters (int) number of polynomials to simulate
    
    """

    
    fig = plt.figure(figsize=(8,8))
    x = np.linspace(-1,1,100) # used for plotting P(x)
    X = [] # to store the real part of the roots
    Y = [] # to store the imaginary part of the roots
    dX = []
    dY = []
    for i in range(niters):
        P = p_type(coeffs,basis)
        R = P.roots() # get the roots of P(x)
        X.append(R.real)
        Y.append(R.imag)
        dP = P.deriv(m=dx)
        dR = dP.roots()
        dX.append(dR.real)
        dY.append(dR.imag)

    plt.scatter(X,Y,s=10,alpha=.7)
    if show_dx_roots:
        plt.scatter(dX,dY,s=10,alpha=.7,color='r')
    plt.title("The Roots of P(x), degree %s" % (len(coeffs)-1))
    plt.xlabel('real')
    plt.ylabel('imag')
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.show()
    
    
    
def deriv_roots_animation():
    params = [('normal',[0,1],{'size':100})]
    coeffs = random_coeffs(params)

    # try also with chebyshev basis

    X,Y,dX,dY = poly_roots(coeffs,basis='power',dx=100,return_values=True)

    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()

    plt.xlim((-1,1))
    plt.ylim((-1,1))

    line, = plt.plot([], [], 'ro', alpha=.4)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return (line,)

    # animation function. This is called sequentially
    def animate(i):
        x = dX[i]
        y = dY[i]
        line.set_data(x, y)
        return (line,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=50)#, blit=True)

    return HTML(anim.to_html5_video())