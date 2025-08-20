import numpy as np
import scipy
from scipy.special import expit






class BaseSmoothOracle(object):
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
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
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

    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

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


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """

    n = x.size
    perturbations = eps * np.eye(n)
    X_perturbed = x[None, :] + perturbations
    f_vec = np.vectorize(func, signature='(n)->()')
    f_orig = func(x)
    f_vals = f_vec(X_perturbed)
    grad = (f_vals - f_orig) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """

    n = x.size
    f_orig = func(x)
    f_vec = np.vectorize(func, signature='(n)->()')
    X1 = x[None, :] + eps * np.eye(n)
    f_x_eps = f_vec(X1)
    Ei = np.eye(n)[:, None, :]
    Ej = np.eye(n)[None, :, :]
    X2 = x[None, None, :] + eps * (Ei + Ej)
    f_x_eps2 = f_vec(X2)
    hess = (f_x_eps2 - f_x_eps[:, None] - f_x_eps[None, :] + f_orig) / (eps ** 2)
    return hess


def lasso_duality_gap(x, Ax_b, ATAx_b, b, lam):
#     """
#     Estimates f(x) - f* via duality gap for 
#         f(x) := 0.5 * ||Ax - b||_2^2 + lam * ||x||_1.
#     """

    norm = np.linalg.norm(ATAx_b, ord=np.inf)
    if norm != 0:
        scal = min(1.0, lam / norm)
    else:
        scal = 1.0
    m = scal * Ax_b
    p = 0.5 * np.dot(Ax_b, Ax_b) + lam * np.linalg.norm(x, ord=1)
    d = -0.5 * np.dot(m, m) - np.dot(b, m)
    return p - d 



class LASSOOptOracle(BaseSmoothOracle):
    """
    Oracle for Lasso barrier method optimization.
    Implements the barrier-approximated function:
        phi_t(x, u) = t*(0.5*||Ax-b||^2 + reg_coef*sum(u)) - sum(log(u+x) + log(u-x))
    """
    def __init__(self, A, b, reg_coef, t):
        self.A = A
        self.b = b
        self.reg_coef = reg_coef
        self.t = t
        
    def func(self, xu):
        n = len(xu) // 2
        x, u = xu[:n], xu[n:]
        residual = self.A.dot(x) - self.b
        f_val = 0.5 * np.sum(residual**2) + self.reg_coef * np.sum(u)
        if np.any(u + x <= 0) or np.any(u - x <= 0):
            return np.inf

        barrier = -np.sum(np.log(u + x)) - np.sum(np.log(u - x))
        return self.t * f_val + barrier
    
    def grad(self, xu):
        n = len(xu) // 2
        x, u = xu[:n], xu[n:]
        residual = self.A.dot(x) - self.b
        grad_x = self.t * (self.A.T.dot(residual)) - (1/(u + x) - 1/(u - x))
        grad_u = self.t * self.reg_coef * np.ones(n) - (1/(u + x) + 1/(u - x))
        return np.concatenate([grad_x, grad_u])
    
    def hess(self, xu):
        n = len(xu) // 2
        x, u = xu[:n], xu[n:]
        a = 1 / (u + x)**2
        b = 1 / (u - x)**2
        H_xx = self.t * (self.A.T.dot(self.A)) + np.diag(a + b)
        H_xu = np.diag(a - b)
        H_uu = np.diag(a + b)
        return np.block([[H_xx, H_xu], [H_xu, H_uu]])


