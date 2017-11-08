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



def animate_roots(P,basis='power'):
    #params = [('beta',[.5,.5],{'size':25}), ('gamma',[2,2],{'size':25}), ('normal',[0,1],{'size':25})]
    #params = [('normal',[0,1],{'size':25}), ('gamma',[2,2],{'size':25}), ('beta',[.5,.5],{'size':25})]
    #params = [('gamma',[2,2],{'size':100})]
    #params = [('normal',[0,1],{'size':100})]

    # try also with chebyshev basis

    X,Y,dX,dY = polyrand.poly_roots(params,basis='power',dx=100,return_values=True)

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
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=100, blit=True)

    HTML(anim.to_html5_video())
