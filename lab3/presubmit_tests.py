import pytest
import numpy as np
import scipy
import scipy.sparse
import scipy.optimize
import sys
import warnings

import optimization
import oracles

# def test_python3():
#     assert sys.version_info > (3, 0)


# def test_least_squares_oracle():
#     A = np.eye(3)
#     b = np.array([1, 2, 3])

#     matvec_Ax = lambda x: A.dot(x)
#     matvec_ATx = lambda x: A.T.dot(x)
#     oracle = oracles.LeastSquaresOracle(matvec_Ax, matvec_ATx, b)

#     # Checks at point x = [0, 0, 0]
#     x = np.zeros(3)
#     assert oracle.func(x) == pytest.approx(7.0)
#     assert np.allclose(oracle.grad(x), np.array([-1., -2., -3.]))
#     assert isinstance(oracle.grad(x), np.ndarray)

#     # Checks at point x = [1, 1, 1]
#     x = np.ones(3)
#     assert oracle.func(x) == pytest.approx(2.5)
#     assert np.allclose(oracle.grad(x), np.array([0., -1., -2.]))
#     assert isinstance(oracle.grad(x), np.ndarray)


# def test_least_squares_oracle_2():
#     A = np.array([[1.0, 2.0], [3.0, 4.0]])
#     b = np.array([1.0, -1.0])

#     matvec_Ax = lambda x: A.dot(x)
#     matvec_ATx = lambda x: A.T.dot(x)
#     oracle = oracles.LeastSquaresOracle(matvec_Ax, matvec_ATx, b)

#     # Checks at point x = [1, 2]
#     x = np.array([1.0, 2.0])
#     assert oracle.func(x) == pytest.approx(80.0)
#     assert np.allclose(oracle.grad(x), np.array([40., 56.]))
#     assert isinstance(oracle.grad(x), np.ndarray)


# def test_l1_reg_oracle():
#     # h(x) = 1.0 * \|x\|_1
#     oracle = oracles.L1RegOracle(1.0)

#     # Checks at point x = [0, 0, 0]
#     x = np.zeros(3)
#     assert oracle.func(x) == pytest.approx(0.0)
#     assert np.allclose(oracle.prox(x, alpha=1.0), x)
#     assert np.allclose(oracle.prox(x, alpha=2.0), x)
#     assert isinstance(oracle.prox(x, alpha=1.0), np.ndarray)

#     # Checks at point x = [-3]
#     x = np.array([-3.0])
#     assert oracle.func(x) == pytest.approx(3.0)
#     assert np.allclose(oracle.prox(x, alpha=1.0), np.array([-2.0]))
#     assert np.allclose(oracle.prox(x, alpha=2.0), np.array([-1.0]))
#     assert isinstance(oracle.prox(x, alpha=1.0), np.ndarray)

#     # Checks at point x = [-3, 3]
#     x = np.array([-3.0, 3.0])
#     assert oracle.func(x) == pytest.approx(6.0)
#     assert np.allclose(oracle.prox(x, alpha=1.0), np.array([-2.0, 2.0]))
#     assert np.allclose(oracle.prox(x, alpha=2.0), np.array([-1.0, 1.0]))
#     assert isinstance(oracle.prox(x, alpha=1.0), np.ndarray)


# def test_l1_reg_oracle_2():
#     # h(x) = 2.0 * \|x\|_1
#     oracle = oracles.L1RegOracle(2.0)

#     # Checks at point x = [-3, 3]
#     x = np.array([-3.0, 3.0])
#     assert oracle.func(x) == pytest.approx(6 * 2.0)
#     assert np.allclose(oracle.prox(x, alpha=1.0), np.array([-1.0, 1.0]))


# def test_lasso_duality_gap():
#     A = np.eye(3)
#     b = np.array([1.0, 2.0, 3.0])
#     regcoef = 2.0

#     # Checks at point x = [0, 0, 0]
#     x = np.zeros(3)
#     assert pytest.approx(0.77777777777777) == oracles.lasso_duality_gap(
#         x, A.dot(x) - b, A.T.dot(A.dot(x) - b), b, regcoef)

#     # Checks at point x = [1, 1, 1]
#     x = np.ones(3)
#     assert pytest.approx(3.0) == oracles.lasso_duality_gap(
#         x, A.dot(x) - b, A.T.dot(A.dot(x) - b), b, regcoef)


# def test_lasso_prox_oracle():
#     A = np.eye(2)
#     b = np.array([1.0, 2.0])
#     oracle = oracles.create_lasso_prox_oracle(A, b, regcoef=1.0)

#     # Checks at point x = [-3, 3]
#     x = np.array([-3.0, 3.0])
#     assert oracle.func(x) == pytest.approx(14.5)
#     assert np.allclose(oracle.grad(x), np.array([-4., 1.]))
#     assert isinstance(oracle.grad(x), np.ndarray)
#     assert np.allclose(oracle.prox(x, alpha=1.0), np.array([-2.0, 2.0]))
#     assert np.allclose(oracle.prox(x, alpha=2.0), np.array([-1.0, 1.0]))
#     assert isinstance(oracle.prox(x, alpha=1.0), np.ndarray)
#     assert oracle.duality_gap(x) == pytest.approx(14.53125)


# def test_lasso_nonsmooth_oracle():
#     A = np.eye(2)
#     b = np.array([1.0, 2.0])
#     oracle = oracles.create_lasso_nonsmooth_oracle(A, b, regcoef=2.0)

#     # Checks at point x = [1, 0]
#     x = np.array([-3.0, 0.0])
#     assert oracle.func(x) == pytest.approx(16.0)
#     assert oracle.duality_gap(x) == pytest.approx(14.5)
#     # Checks a subgradient
#     g = oracle.subgrad(x)
#     assert isinstance(g, np.ndarray)
#     assert g[0] == pytest.approx(-6.0)
#     assert g[1] == pytest.approx(-2.0)


# def check_prototype_results(results, groundtruth):
#     if groundtruth[0] is not None:
#         assert np.allclose(np.array(results[0]), np.array(groundtruth[0]))

#     if groundtruth[1] is not None:
#         assert results[1] == groundtruth[1]

#     if groundtruth[2] is not None:
#         assert results[2] is not None
#         assert 'time' in results[2]
#         assert 'func' in results[2]
#         assert 'duality_gap' in results[2]
#         assert len(results[2]['func']) == len(groundtruth[2])
#     else:
#         assert results[2] is None


# def test_barrier_prototype():
#     method = optimization.barrier_method_lasso
#     A = np.eye(2)
#     b = np.array([1.0, 2.0])
#     reg_coef = 2.0
#     x_0 = np.array([10.0, 10.0])
#     u_0 = np.array([11.0, 11.0])
#     ldg = oracles.lasso_duality_gap

#     method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg)
#     check_prototype_results(
#         method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, tolerance=1e10),
#         [(x_0, u_0), 'success', None])
#     check_prototype_results(
#         method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, tolerance=1e10, trace=True),
#         [(x_0, u_0), 'success', [0.0]])
#     check_prototype_results(
#         method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, max_iter=1, trace=True),
#         [None, 'iterations_exceeded', [0.0, 0.0]])
#     method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, tolerance_inner=1e-8)
#     method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, max_iter=1)
#     method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, max_iter_inner=1)
#     method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, t_0=1)
#     method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, gamma=10)
#     method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, c1=1e-4)
#     method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, trace=True)
#     method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, display=True)
#     method(A, b, reg_coef, x_0, u_0, 1e-5, 1e-8, 100, 20, 1, 10, 1e-4, ldg, True, True)


# def test_subgradient_prototype():
#     method = optimization.subgradient_method

#     A = np.array([[1.0, 2.0], [3.0, 4.0]])
#     b = np.array([1.0, 2.0])
#     oracle = oracles.create_lasso_nonsmooth_oracle(A, b, regcoef=2.0)
#     x_0 = np.array([-3.0, 0.0])

#     method(oracle, x_0)
#     check_prototype_results(
#         method(oracle, x_0, tolerance=1e10),
#         [x_0, 'success', None])
#     check_prototype_results(
#         method(oracle, x_0, tolerance=1e10, trace=True),
#         [None, 'success', [0.0]])
#     check_prototype_results(
#         method(oracle, x_0, max_iter=1),
#         [None, 'iterations_exceeded', None])
#     check_prototype_results(
#         method(oracle, x_0, max_iter=1, trace=True),
#         [None, 'iterations_exceeded', [0.0, 0.0]])
#     method(oracle, x_0, alpha_0=1)
#     method(oracle, x_0, display=True)
#     method(oracle, x_0, 1e-2, 100, 1, True, True)


# def test_proximal_gd_prototype():
#     method = optimization.proximal_gradient_descent

#     A = np.array([[1.0, 2.0], [3.0, 4.0]])
#     b = np.array([1.0, 2.0])
#     oracle = oracles.create_lasso_prox_oracle(A, b, regcoef=2.0)
#     x_0 = np.array([-3.0, 0.0])

#     method(oracle, x_0)
#     method(oracle, x_0, L_0=1)
#     check_prototype_results(
#         method(oracle, x_0, tolerance=1e10),
#         [None, 'success', None])
#     check_prototype_results(
#         method(oracle, x_0, tolerance=1e10, trace=True),
#         [None, 'success', [0.0]])
#     check_prototype_results(
#         method(oracle, x_0, max_iter=1),
#         [None, 'iterations_exceeded', None])
#     check_prototype_results(
#         method(oracle, x_0, max_iter=1, trace=True),
#         [None, 'iterations_exceeded', [0.0, 0.0]])
#     method(oracle, x_0, display=True)
#     method(oracle, x_0, 1, 1e-5, 100, True, True)


# def test_subgradient_one_step():
#     # Simple smooth quadratic task.
#     A = np.eye(2)
#     b = np.array([1.0, 0.0])
#     oracle = oracles.create_lasso_nonsmooth_oracle(A, b, regcoef=0.0)
#     x_0 = np.zeros(2)

#     [x_star, status, hist] = optimization.subgradient_method(
#         oracle, x_0, trace=True)
#     assert status == 'success'
#     assert np.allclose(x_star, np.array([1.0, 0.0]))
#     assert np.allclose(np.array(hist['func']), np.array([0.5, 0.0]))


# def test_subgradient_one_step_nonsmooth():
#     # Minimize 0.5 * ||x - b||_2^2 + ||x||_1
#     # with small tolerance by one step.
#     A = np.eye(2)
#     b = np.array([3.0, 3.0])
#     oracle = oracles.create_lasso_nonsmooth_oracle(A, b, regcoef=1.0)
#     x_0 = np.ones(2)
#     [x_star, status, hist] = optimization.subgradient_method(
#         oracle, x_0, tolerance=1e-1, trace=True)
#     assert status == 'success'
#     assert np.allclose(x_star, np.array([1.70710678, 1.70710678]))
#     assert np.allclose(np.array(hist['func']), np.array([6.0, 5.085786437626]))


# def test_proximal_gd_one_step():
#     # Simple smooth quadratic task.
#     A = np.eye(2)
#     b = np.array([1.0, 0.0])
#     oracle = oracles.create_lasso_prox_oracle(A, b, regcoef=0.0)
#     x_0 = np.zeros(2)

#     [x_star, status, hist] = optimization.proximal_gradient_descent(
#         oracle, x_0, trace=True)
#     assert status == 'success'
#     assert np.allclose(x_star, np.array([1.0, 0.0]))
#     assert np.allclose(np.array(hist['func']), np.array([0.5, 0.0]))


# def test_proximal_nonsmooth():
#     # Minimize ||x||_1.
#     oracle = oracles.create_lasso_prox_oracle(np.zeros([2, 2]),
#                                             np.zeros(2),
#                                             regcoef=1.0)
#     x_0 = np.array([2.0, -1.0])
#     [x_star, status, hist] = optimization.proximal_gradient_descent(
#         oracle, x_0, trace=True)
#     assert status == 'success'
#     assert np.allclose(x_star, np.array([0.0, 0.0]))
#     assert np.allclose(np.array(hist['func']), np.array([3.0, 1.0, 0.0]))



import pytest
import numpy as np
import scipy
from oracles import QuadraticOracle, LASSOOptOracle, lasso_duality_gap
from optimization import barrier_method_lasso, newton, LineSearchTool

def test_quadratic_oracle():
    A = np.eye(3)
    b = np.array([1, 2, 3])
    oracle = QuadraticOracle(A, b)

    # Test at x = [0, 0, 0]
    x = np.zeros(3)
    assert oracle.func(x) == pytest.approx(0.0)
    assert np.allclose(oracle.grad(x), -b)
    assert np.allclose(oracle.hess(x), A)

    # Test at x = [1, 1, 1]
    x = np.ones(3)
    assert oracle.func(x) == pytest.approx(0.5 * np.sum(A) - np.sum(b))
    assert np.allclose(oracle.grad(x), A.dot(x) - b)
    assert np.allclose(oracle.hess(x), A)

def test_lasso_duality_gap():
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])
    regcoef = 2.0

    # Test at x = [0, 0, 0]
    x = np.zeros(3)
    Ax_b = A.dot(x) - b
    ATAx_b = A.T.dot(Ax_b)
    assert lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef) == pytest.approx(7/9)

    # Test at x = [1, 1, 1]
    x = np.ones(3)
    Ax_b = A.dot(x) - b
    ATAx_b = A.T.dot(Ax_b)
    assert lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef) == pytest.approx(3.0)

def test_barrier_method_lasso():
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    reg_coef = 0.1
    x_0 = np.array([0.5, 0.5])
    u_0 = np.array([1.0, 1.0])
    
    (x_star, u_star), message, history = barrier_method_lasso(
        A, b, reg_coef, x_0, u_0, 
        lasso_duality_gap=lasso_duality_gap,
        tolerance=1e-5,
        max_iter=100
    )
    
    assert message == 'success'
    assert np.allclose(x_star, [0.9, 1.9], atol=1e-2)
    assert np.allclose(u_star, [0.9, 1.9], atol=1e-2)

def test_newton_method():
    A = 2 * np.eye(2)
    b = np.array([2.0, 4.0])
    oracle = QuadraticOracle(A, b)
    x_0 = np.array([0.0, 0.0])
    
    x_star, message, history = newton(
        oracle, 
        x_0, 
        tolerance=1e-5,
        line_search_options={'method': 'Armijo'},
        display=False
    )
    
    assert message == 'success'
    assert np.allclose(x_star, [1.0, 2.0], atol=1e-5)


def test_lasso_opt_oracle():
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    reg_coef = 1.0
    t = 1.0
    oracle = LASSOOptOracle(A, b, reg_coef, t)
    
    xu = np.array([0.5, 0.5, 1.0, 1.0])
    func_val = oracle.func(xu)
    grad_val = oracle.grad(xu)
    hess_val = oracle.hess(xu)
    
    assert func_val > 0
    assert grad_val.shape == (4,)
    assert hess_val.shape == (4, 4)

def test_line_search_tool():
    A = np.eye(2)
    b = np.array([1.0, 0.0])
    oracle = QuadraticOracle(A, b)
    x_k = np.array([0.0, 0.0])
    d_k = np.array([1.0, 0.0])
    
    # # Test constant step size
    # ls_tool = LineSearchTool(method='Constant', c=0.5)
    # alpha = ls_tool.line_search(oracle, x_k, d_k)
    # assert alpha == 0.5
    
    # Test Armijo
    ls_tool = LineSearchTool(method='Armijo', c1=1e-4, alpha_0=1.0)
    alpha = ls_tool.line_search(oracle, x_k, d_k)
    assert alpha > 0
    
    # # Test Wolfe
    # ls_tool = LineSearchTool(method='Wolfe', c1=1e-4, c2=0.9)
    # alpha = ls_tool.line_search(oracle, x_k, d_k)
    # assert alpha > 0


# # def test_compute_alpha_max():
# #     # Тест 1: Нет ограничений (alpha_max = inf)
# #     xu = np.array([1.0, 1.0, 2.0, 2.0])
# #     d = np.array([0.1, 0.1, 0.1, 0.1])
# #     assert compute_alpha_max(xu, d) == np.inf
    
# #     # Тест 2: Одно ограничение активно
# #     xu = np.array([1.0, 1.0, 2.0, 2.0])
# #     d = np.array([-0.5, -0.5, -0.5, -0.5])  # Направление, нарушающее ограничение
# #     alpha_max = compute_alpha_max(xu, d)
# #     assert pytest.approx(alpha_max) == 3.0
    
# #     # Тест 3: Несколько ограничений
# #     xu = np.array([0.5, 0.5, 1.0, 1.0])
# #     d = np.array([-0.2, -0.1, -0.3, -0.4])
# #     alpha_max = compute_alpha_max(xu, d)
# #     assert pytest.approx(alpha_max) == min(1.5/0.5, 0.5/0.3)  # min((0.5+1)/(0.2+0.3), (1-0.5)/(0.4-(-0.1)))

# # def test_is_feasible():
# #     # Допустимая точка
# #     xu = np.array([0.1, -0.2, 0.5, 0.5])
# #     assert is_feasible(xu) == True
    
# #     # Недопустимая точка (u < |x|)
# #     xu = np.array([0.6, -0.7, 0.5, 0.5])
# #     assert is_feasible(xu) == False
    
# #     # Граничный случай
# #     xu = np.array([0.5, -0.5, 0.5, 0.5])
# #     assert is_feasible(xu) == False  # Должно быть строгое неравенство

# def test_barrier_method_edge_cases():
#     A = np.eye(2)
#     b = np.array([1.0, 2.0])
#     reg_coef = 0.1
    
#     # Тест 1: Начальная точка недопустима
#     x_0 = np.array([2.0, 2.0])
#     u_0 = np.array([1.0, 1.0])  # u_i < |x_i|
#     (x_star, u_star), message, _ = barrier_method_lasso(
#         A, b, reg_coef, x_0, u_0, 
#         lasso_duality_gap=lasso_duality_gap
#     )
#     assert message == "initial_point_infeasible"
    
#     # Тест 2: Превышено число итераций
#     x_0 = np.array([0.5, 0.5])
#     u_0 = np.array([1.0, 1.0])
#     (x_star, u_star), message, _ = barrier_method_lasso(
#         A, b, reg_coef, x_0, u_0, 
#         lasso_duality_gap=lasso_duality_gap,
#         max_iter=1
#     )
#     assert message == "iterations_exceeded"
    
#     # Тест 3: Нулевая матрица
#     A_zero = np.zeros((2, 2))
#     # (x_star, u_star), message, _ = barrier_method_lasso(
#     #     A_zero, b, reg_coef, x_0, u_0, 
#     #     lasso_duality_gap=lasso_duality_gap
#     # )
#     (x_star, u_star), message, _ = barrier_method_lasso(
#         A_zero, b, reg_coef,
#         x_0=np.array([0.0, 0.0]),
#         u_0=np.array([0.5, 0.5]),  # ближе к границе
#         lasso_duality_gap=lasso_duality_gap,
#         t_0=100,                  # сразу большой t (меньше влияние барьера)
#         gamma=50,                 # быстрее усиливаем t
#         max_iter=30               # достаточно итераций
#     )
#     # Проверяем, что решение стремится к 0
#     assert np.allclose(x_star, [0, 0], atol=1e-2)
#     # assert np.allclose(u_star, [0, 0], atol=1e-2)
#     assert np.linalg.norm(u_star) < 1.0

# def test_newton_method_edge_cases():
#     # Тест 1: Неположительно определенная гессиана
#     A = -np.eye(2)  # Отрицательно определенная матрица
#     b = np.array([1.0, 1.0])
#     oracle = QuadraticOracle(A, b)
#     x_0 = np.array([0.0, 0.0])
    
#     x_star, message, _ = newton(
#         oracle, 
#         x_0, 
#         tolerance=1e-5,
#         display=False
#     )
#     assert message == 'newton_direction_error'
    
#     # Тест 2: Вырожденная гессиана
#     A = np.array([[1.0, 1.0], [1.0, 1.0]])  # Вырожденная матрица
#     b = np.array([1.0, 1.0])
#     oracle = QuadraticOracle(A, b)
#     x_0 = np.array([0.0, 0.0])
    
#     x_star, message, _ = newton(
#         oracle, 
#         x_0, 
#         tolerance=1e-5,
#         display=False
#     )
#     assert message == 'newton_direction_error'
    
#     # Тест 3: Начальная точка с NaN в градиенте
#     class BadOracle(QuadraticOracle):
#         def grad(self, x):
#             return np.array([np.nan, np.nan])
    
#     oracle = BadOracle(np.eye(2), np.zeros(2))
#     x_0 = np.array([0.0, 0.0])
    
#     x_star, message, _ = newton(
#         oracle, 
#         x_0, 
#         tolerance=1e-5,
#         display=False
#     )
#     assert message == 'computational_error'

# def test_lasso_duality_gap_edge_cases():
#     # Тест 1: Нулевой вектор ATAx_b
#     A = np.zeros((3, 3))
#     b = np.array([1.0, 2.0, 3.0])
#     x = np.zeros(3)
#     Ax_b = A.dot(x) - b
#     ATAx_b = A.T.dot(Ax_b)
#     gap = lasso_duality_gap(x, Ax_b, ATAx_b, b, 1.0)
#     assert gap >= 0
    
#     # Тест 2: Большой коэффициент регуляризации
#     A = np.eye(3)
#     b = np.array([1.0, 2.0, 3.0])
#     x = np.ones(3)
#     Ax_b = A.dot(x) - b
#     ATAx_b = A.T.dot(Ax_b)
#     gap = lasso_duality_gap(x, Ax_b, ATAx_b, b, 100.0)
#     assert gap > 10.0  # Должен быть большим

# def test_lasso_opt_oracle_edge_cases():
#     # Тест 1: Точка на границе допустимой области
#     A = np.eye(2)
#     b = np.array([1.0, 2.0])
#     reg_coef = 1.0
#     t = 1.0
#     oracle = LASSOOptOracle(A, b, reg_coef, t)
    
#     xu = np.array([0.999, 0.999, 1.0, 1.0])  # Почти на границе
#     func_val = oracle.func(xu)
#     grad_val = oracle.grad(xu)
#     hess_val = oracle.hess(xu)
    
#     assert np.isfinite(func_val)
#     assert np.all(np.isfinite(grad_val))
#     assert np.all(np.isfinite(hess_val))
# def test_line_search_tool_edge_cases():
#     # Тест 1: Направление нулевого градиента      
#     A = np.eye(2)
#     b = np.array([1.0, 0.0])
#     oracle = QuadraticOracle(A, b)
#     x_k = np.array([1.0, 0.0])  # Оптимальная точка
#     d_k = np.array([0.0, 0.0])

#     ls_tool = LineSearchTool(method='Armijo')     
#     alpha = ls_tool.line_search(oracle, x_k, d_k) 
#     assert alpha == 1.0  # Должен вернуть начальный шаг

#     # Тест 2: Метод Вульфа на гладкой выпуклой функции  
#     class SmoothConvexOracle:
#         def func_directional(self, x, d, alpha):
#             return (alpha - 0.5)**2  # Минимум в alpha=0.5

#         def grad_directional(self, x, d, alpha):
#             return 2 * (alpha - 0.5)

#     oracle = SmoothConvexOracle()
#     x_k = np.array([0.0])
#     d_k = np.array([1.0])

#     ls_tool = LineSearchTool(method='Wolfe', c1=1e-4, c2=0.9)
#     alpha = ls_tool.line_search(oracle, x_k, d_k)
#     assert alpha is not None
#     assert 0.4 < alpha < 0.6  # Ожидаем, что шаг около 0.5


if __name__ == "__main__":
    pytest.main([__file__])