from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
from time import time
import datetime
import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict
from oracles import QuadraticOracle
from scipy.optimize import line_search
from scipy.optimize._linesearch import scalar_search_wolfe2
from oracles import LASSOOptOracle



class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented
            for computing function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
            previous steps. If None, self.alpha_0 is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """


        if self._method == 'Constant':
            return self.c

        phi = lambda alpha: oracle.func_directional(x_k, d_k, alpha)
        derphi = lambda alpha: oracle.grad_directional(x_k, d_k, alpha)
        phi0 = phi(0)
        derphi0 = derphi(0)

        if self._method == 'Wolfe':
            result = scalar_search_wolfe2(
                phi=phi,
                derphi=derphi,
                phi0=phi0,
                derphi0=derphi0,
                c1=self.c1,
                c2=self.c2
            )
            alpha = result[0] if result is not None else None
            if alpha is not None:
                return alpha

        alpha_max = compute_alpha_max(x_k, d_k)
        # print(self.alpha_0, 7878787873434343434)
        if alpha_max == 1:
            return 1
        if alpha_max == 0:
            return self.alpha_0 
        alpha = min(self.alpha_0, 0.99 * alpha_max)
        max_iter = 20
        i = 0
        while phi(alpha) > phi(0) + self.c1 * alpha * derphi(0):
            alpha /= 2
            i += 1
            if i > max_iter or alpha < 1e-8:
                return self.alpha_0
        # if alpha == 0:
        #     return None
        return alpha
    



def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()
    
def compute_alpha_max(xu, d):
    n = len(xu) // 2
    x, u = xu[:n], xu[n:]
    dx, du = d[:n], d[n:]
    mask1 = (dx + du) < 0
    mask2 = (du - dx) < 0

    alpha1 = np.full(n, np.inf)
    alpha2 = np.full(n, np.inf)
    alpha1[mask1] = -(x[mask1] + u[mask1]) / (dx[mask1] + du[mask1])
    alpha2[mask2] = -(u[mask2] - x[mask2]) / (du[mask2] - dx[mask2])
    if alpha1.size + alpha2.size == 0:
            return 1
    return np.min(np.hstack([alpha1, alpha2]))

# def is_feasible(xu):
#     n = len(xu) // 2
#     x, u = xu[:n], xu[n:]
#     return np.all(u > np.abs(x))

def _log_history(history, x_k, u_k, gap, start_time, A, b, reg_coef):

    Ax_b = A.dot(x_k) - b
    f_val = 0.5 * np.sum(Ax_b**2) + reg_coef * np.sum(u_k)
    history['time'].append((datetime.now() - start_time).total_seconds())
    history['func'].append(f_val)
    history['duality_gap'].append(gap)
    if x_k.size <= 2:
        history['x'].append(x_k.copy())


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5, 
                         tolerance_inner=1e-8, max_iter=100, 
                         max_iter_inner=20, t_0=1, gamma=10, 
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary 
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    n = x_0.size
    history = defaultdict(list) if trace else None
    message = 'success'
    x_k, u_k = np.copy(x_0), np.copy(u_0)
    t_k = t_0

    if not np.all(u_0 > np.abs(x_0)):
        return (x_0, u_0), 'initial_point_infeasible', None

    Ax_b_fun = lambda x: A.dot(x) - b
    ATAx_b_fun = lambda x: A.T.dot(A.dot(x) - b)

    start_time = datetime.now()
    current_Ax_b = Ax_b_fun(x_k)
    current_ATAx_b = ATAx_b_fun(x_k)
    gap = lasso_duality_gap(x_k, current_Ax_b, current_ATAx_b, b, reg_coef)
    if trace:
        _log_history(history, x_k, u_k, gap, start_time, A, b, reg_coef)
    if gap <= tolerance:
        return (x_k, u_k), 'success', history
    for k in range(max_iter):
        oracle = LASSOOptOracle(A, b, reg_coef, t_k)
        xu0 = np.hstack([x_k, u_k])
        xu_opt, msg, _ = newton(oracle, xu0, tolerance_inner, max_iter_inner)
        if msg != 'success':
            message = 'computational_error'
            break
        x_k, u_k = xu_opt[:n], xu_opt[n:]
        t_k *= gamma
        current_Ax_b = Ax_b_fun(x_k)
        current_ATAx_b = ATAx_b_fun(x_k)
        gap = lasso_duality_gap(x_k, current_Ax_b, current_ATAx_b, b, reg_coef)
        if trace:
            _log_history(history, x_k, u_k, gap, start_time, A, b, reg_coef)
        if display:
            f_val = 0.5*np.sum(current_Ax_b**2) + reg_coef*np.sum(u_k)
            print(f"Iter {k}: t={t_k:.1e}, gap={gap:.3e}, f={f_val:.3e}")
        if gap <= tolerance:
            break
    else:
        message = 'iterations_exceeded'

    return (x_k, u_k), message, history



def newton(oracle, x_0, tolerance=1e-5, max_iter=100, #Задание 6
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    g0 = oracle.grad(x_k)
    if g0 is None or np.any(np.isnan(g0)) or np.any(np.isinf(g0)):
        return x_k, 'computational_error', history
    norm_grad0_sq = np.linalg.norm(g0)**2
    if np.isnan(norm_grad0_sq) or np.isinf(norm_grad0_sq):
        return x_k, 'computational_error', history
    if trace:
        history['time'].append(0.0)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.linalg.norm(g0))
        if x_k.size <= 2:
            history['x'].append(x_k.copy())
    start_time = datetime.now()
    alpha = None
    for k in range(max_iter):
        g_k = oracle.grad(x_k)
        if g_k is None or np.any(np.isnan(g_k)) or np.any(np.isinf(g_k)):
            return x_k, 'computational_error', history
        norm_grad_sq = np.linalg.norm(g_k)**2
        if np.isnan(norm_grad_sq) or np.isinf(norm_grad_sq):
            return x_k, 'computational_error', history
        if norm_grad_sq <= tolerance * norm_grad0_sq:
            if display:
                print("Остановка по критерию: ||grad(x)||^2 = {} <= {} * ||grad(x0)||^2".format(norm_grad_sq, tolerance))
            return x_k, 'success', history

        H_k = oracle.hess(x_k)
        try:
            cho_factor = scipy.linalg.cho_factor(H_k)
            d_k = scipy.linalg.cho_solve(cho_factor, -g_k)
        except LinAlgError as e:
            if display:
                print("Ошибка при вычислении направления Ньютона (LinAlgError): {}".format(e))
            return x_k, 'newton_direction_error', history

        if d_k is None or not np.all(np.isfinite(d_k)):
            if display:
                print("Найденное направление не является конечным!")
            return x_k, 'computational_error', history
        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        if display:
            print("Итерация {:3d}: f(x) = {:.6f}, ||grad(x)|| = {:.6e}, α = {:.6f}".format(
                  k, oracle.func(x_k), np.linalg.norm(g_k), alpha))
        x_k = x_k + alpha * d_k
        norm = np.linalg.norm(oracle.grad(x_k))
        if trace:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(elapsed_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(norm)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
    if norm ** 2 <= tolerance * norm_grad0_sq: 
        return x_k, 'success', history
    return x_k, 'iterations_exceeded', history

