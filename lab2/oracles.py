import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

    def minimize_directional(self, x, d):
        """
        Minimizes the function with respect to a specific direction:
            Finds alpha = argmin f(x + alpha d)
        """
        return (-self.grad(x) @ d) / (np.dot(d, np.dot(self.A, d)))


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATy : function of y
            Computes matrix-vector product A^Ty, where y is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        Ax = self.matvec_Ax(x)
        y = self.b * Ax
        m = self.b.shape[0]
        loss = np.logaddexp(0, -y)  
        return np.mean(loss) + 0.5 * self.regcoef * np.dot(x, x)

    def grad(self, x):
        Ax = self.matvec_Ax(x)
        y = self.b * Ax
        m = self.b.shape[0]
        coeff = -expit(-y)
        coeff = self.b * coeff
        grad_loss = self.matvec_ATx(coeff)
        return grad_loss / m + self.regcoef * x

    def hess(self, x):
        Ax = self.matvec_Ax(x)
        y = self.b * Ax
        m = self.b.shape[0]
        s = expit(-y)
        diag_s = s * (1 - s)
        H_loss = self.matmat_ATsA(diag_s) / m
        n = x.shape[0]
        return H_loss + self.regcoef * np.eye(n)
    def hess_vec(self, x, v):
        Ax = self.matvec_Ax(x)
        y = -self.b * Ax
        sigma = expit(y)
        D = sigma * (1 - sigma)
        Av = self.matvec_Ax(v)
        DAv = D * Av
        At_DAv = self.matvec_ATx(DAv)
        m = self.b.shape[0]
        return At_DAv / m + self.regcoef * v

class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        Ax = self.matvec_Ax(x)
        Ad = self.matvec_Ax(d)
        y = self.b * (Ax + alpha * Ad)
        m = self.b.shape[0]
        loss = np.logaddexp(0, -y)
        reg = 0.5 * self.regcoef * np.dot(x + alpha * d, x + alpha * d)
        return np.mean(loss) + reg

    def grad_directional(self, x, d, alpha):
        Ax = self.matvec_Ax(x)
        Ad = self.matvec_Ax(d)
        y = self.b * (Ax + alpha * Ad)
        m = self.b.shape[0]
        coeff = -expit(-y) * self.b
        inner = np.dot(Ad, coeff) / m
        reg_deriv = np.dot(x + alpha * d, d)
        return inner + self.regcoef * reg_deriv


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Количество образцов в A ({A.shape[0]}) и b ({b.shape[0]}) должно совпадать")
    if regcoef < 0:
        raise ValueError("regcoef должен быть неотрицательным")

    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    def matmat_ATsA(s):
        if scipy.sparse.issparse(A):
            S = scipy.sparse.diags(s)
            return A.T @ S @ A
        else:
            return A.T @ (s[:, None] * A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

def hess_vec_finite_diff(func, x, v, eps=1e-5):
    n = x.size
    Hv = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        Hv[i] = (func(x + eps * v + eps * ei) - func(x + eps * v)
                 - func(x + eps * ei) + func(x)) / (eps ** 2)
    return Hv