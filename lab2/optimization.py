import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool
from datetime import datetime

def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    
    n = b.shape[0]
    if max_iter is None:
        max_iter = 2 * n
    start_time = datetime.now()

    g_k = matvec(x_k) - b
    d_k = -g_k

    b_norm = np.linalg.norm(b)
    if trace:
        history['time'].append(0.0)
        history['residual_norm'].append(np.linalg.norm(g_k))
        if x_k.size <= 2:
            history['x'].append(x_k.copy())

    for k in range(max_iter):
        if np.linalg.norm(g_k) <= tolerance * b_norm:
            if display:
                print(f"Converged at iter {k}, residual {np.linalg.norm(g_k):.2e}")
            return x_k, 'success', history
        
        Ad_k = matvec(d_k)



        alpha_k = np.dot(g_k, g_k) / np.dot(Ad_k, d_k)

        x_k = x_k + alpha_k * d_k
        # считаем так, потому что экономим матрично-веторную операцию, разложили формулу

        g_next = g_k + alpha_k * Ad_k

        beta_k = np.dot(g_next, g_next) / np.dot(g_k, g_k)
        d_k = -g_next + beta_k * d_k
        g_k = g_next
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['residual_norm'].append(np.linalg.norm(g_k))
            if x_k.size <= 2:
                history['x'].append(x_k.copy())

        if display:
            print(f"Iter {k+1}: residual={np.linalg.norm(g_k):.2e}, alpha={alpha_k:.2e}, beta={beta_k:.2e}")

    if display:
        print(f"Max iterations exceeded: {max_iter}, residual={np.linalg.norm(g_k):.2e}")
    return x_k, 'iterations_exceeded', history



def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = x_0.copy()

    grad = oracle.grad(x_k)
    grad_norm = np.linalg.norm(grad)
    criterion = tolerance * (grad_norm ** 2)
    start_time = datetime.now()
    
    if trace:
        history['time'].append(0.0)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(grad_norm)
        if x_k.size <= 2:
            history['x'].append(x_k.copy())

    H = deque(maxlen=memory_size) 
    i = 0
    message = 'success'

    while i < max_iter and grad_norm**2 > criterion:
        if not H:
            d_k = -grad
        else:
            d_k = LBFGS_direction(-grad, H)

        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        if alpha is None:
            message = 'computational_error'
            break

        s_k = alpha * d_k
        x_new = x_k + s_k
        new_grad = oracle.grad(x_new)

        if not np.isfinite(new_grad).all():
            message = 'computational_error'
            break

        y_k = new_grad - grad
        if np.dot(s_k, y_k) > 0:
            H.append((s_k, y_k))

        x_k = x_new
        grad = new_grad
        grad_norm = np.linalg.norm(grad)
        i += 1

        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())

        if display:
            print(f"Iter {i}: grad_norm={grad_norm:.2e}")

    if i == max_iter and grad_norm**2 > criterion:
        message = 'iterations_exceeded'

    return x_k, message, history
def BFGS_Multiply(v, H, gamma_0):
        if not H:
            return gamma_0 * v
        s, y = H[-1] 
        H_prev = deque(list(H)[:-1])
        v_prime = v - (np.dot(s, v) / np.dot(y, s)) * y
        z = BFGS_Multiply(v_prime, H_prev, gamma_0)
        return z + ((np.dot(s, v) - np.dot(y, z)) / np.dot(y, s)) * s


def LBFGS_direction(v, H):
        if H:
            s_last, y_last = H[-1]
            gamma_0 = np.dot(y_last, s_last) / np.dot(y_last, y_last)
        else:
            gamma_0 = 1.0
        return BFGS_Multiply(v, H, gamma_0)




def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = x_0.copy()

    start_time = datetime.now()
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)

    g_k_sqr = g_0_sqr = np.linalg.norm(g_k) **2
    if trace:
        history['func'].append(f_k)
        history['time'].append(0.0)
        history['grad_norm'].append(np.sqrt(g_k_sqr))
        if x_k.size <= 2:
            history['x'].append(x_k.copy())

    for k in range(1, max_iter + 1):
        if g_k_sqr <= tolerance * g_0_sqr:
            if display:
                print(f"Converged at iter {k-1}, grad_norm={np.sqrt(g_k_sqr):.2e}")
            return x_k, 'success', history

        eta_k = min(0.5, np.sqrt(np.sqrt(g_k_sqr)))
        b = -g_k
        d0 = b.copy()

        d_k, _, _ = conjugate_gradients(
            lambda v: oracle.hess_vec(x_k, v),
            b, d0,
            tolerance=eta_k,
            max_iter=None,
            trace=False,
            display=False
        )

        inner = np.dot(g_k, d_k)
        while inner >= 0:
            eta_k *= 0.1
            d0 = b.copy()
            d_k, _, _ = conjugate_gradients(
                lambda v: oracle.hess_vec(x_k, v),
                b, d0,
                tolerance=eta_k,
                max_iter=None,
                trace=False,
                display=False
            )
            inner = np.dot(g_k, d_k)

        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        if alpha_k is None:
            alpha_k = 1e-6

        x_k = x_k + alpha_k * d_k
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        g_k_sqr = np.linalg.norm(g_k) **2

        if trace:
            history['func'].append(f_k)
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['grad_norm'].append(np.sqrt(g_k_sqr))
            if x_k.size <= 2:
                history['x'].append(x_k.copy())

        if display:
            print(f"Iter {k}: f={f_k:.4e}, grad_norm={np.sqrt(g_k_sqr):.2e}, alpha={alpha_k:.2e}, eta={eta_k:.2e}")

    if display:
        print(f"Max iterations exceeded: {max_iter}, grad_norm={np.sqrt(g_k_sqr):.2e}")
    return x_k, 'iterations_exceeded', history




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


