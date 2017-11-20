import rootfinder as rf

def test_frac_deriv():
    coeffs = np.random.randn(5)
    P = np.polynomial.polynomial.Polynomial(coeffs)
    assert np.allclose(frac_deriv(coeffs,0),np.polynomial.polynomial.polycompanion(P.coef))
    assert np.allclose(frac_deriv(coeffs,1),np.polynomial.polynomial.polycompanion(P.deriv(m=1).coef))
    assert np.allclose(frac_deriv(coeffs,2),np.polynomial.polynomial.polycompanion(P.deriv(m=2).coef))


def test_frac_deriv_poly():
    