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

def gen_coeffs(n,dist='normal'):
    if dist == 'uniform':
        coeffs = np.random.uniform(0,1,size=n)
        coeffs[-1] = 1.
        return coeffs
    else:
        coeffs = np.random.randn(n)
        coeffs[-1] = 1.
        return coeffs

def fracdiff(coeffs,a):
    return frac_deriv(coeffs,a,return_coeffs=True)


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
    

def timelapse(coeffs,start,stop,n_frames,basis='power',plot_axis=True,saveas=None,figtitle=None):
    
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
    if figtitle:
        plt.title(figtitle)
    else:
        plt.title("roots of degree %s polynomial after %s derivatives" % (P.degree(),stop-start))
    plt.axis('equal')
    if saveas:
        plt.xlim(-1.2,1.2)
        plt.ylim(-1.2,1.2)
        plt.savefig(saveas,bbox="tight")
        plt.close()
    else:
        plt.show()
    
    

def animate_roots(coeffs,
                  start=None,
                  stop=None,
                  n_frames=200,
                  t_interval=75,
                  plot_trail=True,
                  plot_range=[(-1.5,1.5),(-1.5,1.5)],
                  saveas=None,
                  figtitle=None):
    """
    Animate the continuous deformation of the roots of fractional derivatives.
    
    """
    if start == None:
        start = 0.
    if stop == None:
        stop = len(coeffs) - 2

    fig = plt.figure(figsize=(8,8))
    if figtitle:
        plt.title(figtitle)
    ax = plt.gca()
    
    plt.xlim(plot_range[0])
    plt.ylim(plot_range[1])

    points, = plt.plot([], [], color='r', marker='o', ls='None')

    line, = plt.plot([], [], color='orange', marker='o', ls='None', alpha=.16)
    greyline, = plt.plot([], [], color='grey', marker='o', ls='None', alpha=.04)

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
    if saveas:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='DGM'), bitrate=1800)
        anim.save(saveas,writer)
        plt.close()
    else:
        return HTML(anim.to_html5_video())



def cheb_nodes(a,b,n):
    """
    get the chebyshev nodes between 0 and 1
    
    """
    nodes = []
    for k in range(1,n+1):
        nodes.append(.5*(b+a) + .5*(b-a)*np.cos((2*k-1)*np.pi/(2*n)))
    return np.sort(np.array(nodes))[::-1]


def pm(coeffs):
    """
    Determines the direction (plus or minus) a root should travel from the origin.
    
    """
    n_pos = 0
    s = np.sign(coeffs[::-1])
    # determine number of positive roots
    for i in range(len(coeffs)-1):
        # this indicates a sign change which implies a positive root
        if s[i]*s[i+1] == -1:
            n_pos += 1
    
    # define new s to find number of negative roots
    s = [-s[i] if i%2 == 1 else s[i] for i in range(len(s))]
    
    
def fractional_newton_method(P,x0,a,maxiters=500):
    L = [x0]
    while True:
        x1 = x0 - P(x0)/frac_deriv_poly(P,a,x0)
        L.append(x1)
        if len(L) >= maxiters:
            break
        elif np.allclose(abs(x1-x0),0):
            break
        else:
            x0 = x1
    return np.array(L)

def frac_newton(P,x0,a,maxiters=100):
    while True:
        x1 = x0 - P(x0)/(P.deriv(1)(x0) - P(x0)*a*x0**-1)
    
def newton_method(x0,P,maxiters=100):
    L = [x0]
    for i in range(maxiters):
        x1 = x0 - P(x0)/frac_deriv_poly(P,1,x0)
        L.append(x1)
        x0 = x1
    return np.array(L)

def newton_opt(x0,P,maxiters=100):
    pass

def fractional_newton_opt(x0,P,a,b,maxiters=100):
    pass

def newton_integration(coeffs,dx,n=5,show=False):
    """
    NIRF (Newton Integration Root Finding)
    Use newton's method to integrate the roots of a polynomial up to a different polynomial.
    
    A fazer:
        + find optimal value for dx
        + find optimal partition (optimal n)
        + find good guess for new root value at zero
    
    """
    # define the partition of fractional differentiation terms
    a = np.linspace(0,dx,n+1)
    
    # initialize the coefficients to be used for Descartes Rule of Signs
    signs = np.sign(coeffs)
    
    # Find known roots of a derivative
    D = frac_deriv(coeffs,dx)
    R_ = np.linalg.eig(D)[0]
    Roots = R_[R_.imag >= 0].copy()
    
    # plot the solution we hope to get to and the starting roots
    if show:
        R = np.polynomial.polynomial.Polynomial(coeffs).roots()
        plt.figure(figsize=(8,8))
        plt.scatter(R.real,R.imag,color='k',alpha=.2,s=150)
        plt.scatter(Roots.real,Roots.imag,color='k',label="Derivative Roots")
    
    step = 0 # records how many integer antiderivatives we have taken
    for i in range(n+1):

        # define the function to optimize
        F_coeffs = frac_deriv(coeffs,dx-a[i],return_coeffs=True)
        P = np.polynomial.polynomial.Polynomial(F_coeffs)
        
        # used for fractional newton's method
        #dF_coeffs = frac_deriv(coeffs,dx-a[i]-1.01,return_coeffs=True)
        #dP = np.polynomial.polynomial.Polynomial(dF_coeffs)
        
        # determines whether to add a new root at zero(ish)        
        if (step < np.floor(a[i])):
            # add a positive or negative value depending on Descartes Rule of Signs
            s = np.floor(a[i])
            pm = signs[int(s)]*signs[int(s)+1]
            #Z = [.1001 + .001j, -.1 + .0j]
            Z = [0.01 + 0.j]
            #Z = []
            step = s
        else:
            Z = []

        # Newton's method to find roots of function to optimize
        for r in Roots:

            f = lambda x: P(x)/x**(a[i])
            try:
                z = newton(f,r)
            except:
                try:
                    # used for fractional newton's method
                    dF_coeffs = frac_deriv(coeffs,dx-a[i]-1.05,return_coeffs=True)
                    dP = np.polynomial.polynomial.Polynomial(dF_coeffs)
                    df = lambda x: P.deriv(1)(x)*x**1.05
                    z = newton(P,r,fprime=df,maxiter=150)
                    print("fractional newton converged!")
                except:
                    print("missed one!")
                    continue
            # ignore negative complex region to save on flops
            if z.imag < 0:
                Z.append(np.conjugate(z))
            else:
                Z.append(z)
        
        Roots = np.unique(Z)
        
        if show:
            plt.scatter(Roots.real,Roots.imag,color='orange',alpha=.2)
        
    if show:
        plt.scatter(Roots.real,Roots.imag,color='r',alpha=.5)
        Roots_ = np.conjugate(Roots)
        plt.axis('equal')
        plt.legend()
        plt.show()
        
    return Roots

def animate_frac_roots(P,x0,dx,t_interval=200,plot_range=[(-1,1),(-1,1)]):
    """
    animate fractional newton method
    """
    R = fractional_newton_method(P,x0,dx)
    n_frames = len(R)

    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    
    plt.xlim(plot_range[0])
    plt.ylim(plot_range[1])

    points, = plt.plot([], [], color='r', marker='o', ls='None')

    line, = plt.plot([], [], color='orange', marker='o', ls='None', alpha=.3)
    greyline, = plt.plot([], [], color='grey', marker='o', ls='None', alpha=.2)

    # initialization function: plot the background of each frame
    def init():
        greyline.set_data([],[])
        line.set_data([], [])
        points.set_data([], [])
        return (greyline, line, points,)

    # animation function. This is called sequentially
    def animate(i):

        # make the orange and grey trails
        if i-15 >= 0:
            line.set_data(R[i-15:i].real, R[i-15:i].imag)
            greyline.set_data(R[:i-15].real, R[:i-15].imag)
        else:
            line.set_data(R[:i].real, R[:i].imag)
            
        # get the current roots (red)
        points.set_data(R[i].real, R[i].imag)

        
        return (greyline, line, points,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=t_interval, blit=True)

    return HTML(anim.to_html5_video())
