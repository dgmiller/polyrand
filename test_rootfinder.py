import polyrand.rootfinder as rf
import numpy as np

def test_frac_deriv():
    coeffs = np.random.randn(5)
    P = np.polynomial.polynomial.Polynomial(coeffs)
    assert np.allclose(rf.frac_deriv(coeffs,0),np.polynomial.polynomial.polycompanion(P.coef))
    assert np.allclose(rf.frac_deriv(coeffs,1),np.polynomial.polynomial.polycompanion(P.deriv(m=1).coef))
    assert np.allclose(rf.frac_deriv(coeffs,2),np.polynomial.polynomial.polycompanion(P.deriv(m=2).coef))


def test_frac_antideriv():
    coeffs = np.array([24.,72.,120.,120.])
    const = np.array([7.,6.,10.])
    P = np.polynomial.polynomial.Polynomial(coeffs)
    print(rf.frac_antideriv(coeffs,3,const))
    print(np.polynomial.polynomial.polycompanion(P.integ(m=3,k=const[::-1]).coef))
    assert np.allclose(rf.frac_antideriv(coeffs,3,const),np.polynomial.polynomial.polycompanion(P.integ(m=3,k=const[::-1]).coef))

    
