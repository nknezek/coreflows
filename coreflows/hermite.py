import numpy as _np
import scipy.misc as _ms

def fun(x, l):
    """compute the hermite basis function of degree l at location x

    :param x:
    :param l:
    :return:
    """
    c = _np.zeros(40)
    c[l] = 1.
    return (2 ** l * _ms.factorial(l) * _np.pi ** 0.5) ** -0.5 * _np.exp(-x ** 2 / 2) * _np.polynomial.hermite.hermval(
        x, c)

def sum(x, coeffs, delta_x):
    """compute sum of set of hermite basis functions

    :param x: locations x  - np.array((Nx))
    :param coeffs: coefficients of basis functions  - np.array((Nc))
    :param delta_x: width parameter of basis functions  - [np.array((Nc)) or float]
    :return: function along x  - np.array((Nx))
    """
    out = _np.zeros_like(x)
    for l in range(len(coeffs)):
        out += coeffs[l] * fun(x / delta_x, l)
    return out

def fit_fun(x, fit_c):
    """define the hermite fit function to use

    :param x: locations - np.array((Nx))
    :param fit_c: containing coefficients [:Nc] and width parameter [-1] - np.array((Nc+1))
    :return:
    """
    return sum(x, fit_c[:-1], fit_c[-1])