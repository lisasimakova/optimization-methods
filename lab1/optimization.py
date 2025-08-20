import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict
from oracles import QuadraticOracle
from scipy.optimize import line_search
from scipy.optimize._linesearch import scalar_search_wolfe2
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
        delphi0 = derphi(0)
        if self._method == 'Wolfe':
            result= scalar_search_wolfe2(                 
                phi=phi,
                derphi=derphi,
                phi0=phi0,
                derphi0=delphi0,
                c1=self.c1,
                c2=self.c2
            )
            alpha = result[0] if result is not None else None
            if alpha is not None:
                return alpha

        if self._method == 'Armijo' or self._method == 'Wolfe':
            if previous_alpha is None:
                alpha_0 = self.alpha_0
            else:
                alpha_0 = previous_alpha
            alpha = alpha_0
            while phi(alpha) > (phi0 + alpha * delphi0 * self.c1):
                alpha /= 2
            return alpha
        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
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
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    alpha = None
    if x_k is None or np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)):
        return x_k, 'computational_error', history
    start_time = datetime.now()

    grad_0 = oracle.grad(x_k)
    if grad_0 is None or np.any(np.isnan(grad_0)) or np.any(np.isinf(grad_0)):
        if display:
            print("Ошибка: градиент grad_0 содержит NaN или Inf")
        return x_k, 'computational_error', history

    grad_0_norm_sq = np.linalg.norm(grad_0) ** 2
    if np.isnan(grad_0_norm_sq) or np.isinf(grad_0_norm_sq):
        if display:
            print("Ошибка: ||grad_0||^2 = NaN или Inf")
        return x_k, 'computational_error', history

    if trace:
        history['time'].append(0.0)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.linalg.norm(grad_0))
        if x_k.size <= 2:
            history['x'].append(x_k.copy())
    for i in range(max_iter):
        grad = oracle.grad(x_k)
        grad_norm = np.linalg.norm(grad)

        if grad is None or np.any(np.isnan(grad)) or np.any(np.isinf(grad)) or np.isnan(grad_norm) or np.isinf(grad_norm):
            if display:
                print("Ошибка: градиент на итерации содержит некорректные значения (NaN или Inf)")
            return x_k, 'computational_error', history

        if grad_norm ** 2 <= tolerance * grad_0_norm_sq:
            if display:
                print(f"Остановка: ||grad||^2 = {grad_norm ** 2:.6e} <= {tolerance} * ||grad_0||^2")
            return x_k, 'success', history

        d_k = -grad
        alpha = line_search_tool.line_search(oracle, x_k, d_k, 2 * alpha if alpha else None) 

        if alpha is None or np.isnan(alpha) or np.isinf(alpha):
            if display:
                print("Ошибка: шаг alpha некорректен (None, NaN или Inf)")
            return x_k, 'computational_error', history

        x_k = x_k + alpha * d_k

        if trace:
            elapsed = (datetime.now() - start_time).total_seconds()
            f_val = oracle.func(x_k)

            if f_val is None or np.isnan(f_val) or np.isinf(f_val):
                if display:
                    print("Ошибка: значение функции f(x) некорректно (None, NaN или Inf)")
                return x_k, 'computational_error', history
            grad = oracle.grad(x_k)
            grad_norm = np.linalg.norm(grad)
            history['time'].append(elapsed)
            history['func'].append(f_val)
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display:
            print(f"Iter {i}: f = {f_val:.6f}, ||grad|| = {grad_norm:.6e}, alpha = {alpha:.3e}")
    if grad_norm ** 2 <= tolerance * grad_0_norm_sq: 
        return x_k, 'success', history   
    return x_k, 'iterations_exceeded', history



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


