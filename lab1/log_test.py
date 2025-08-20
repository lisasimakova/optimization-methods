import numpy as np
from oracles import create_log_reg_oracle, grad_finite_diff, hess_finite_diff
from scipy.sparse import csr_matrix

np.random.seed(42)
m, n = 10, 5
A_dense = np.random.randn(m, n)
b = np.random.choice([-1, 1], size=m)
regcoef = 1e-2

A_sparse = csr_matrix(A_dense)

test_points = [np.random.randn(n) for _ in range(3)]

def check_grad_hess(oracle, x):
    analytic_grad = oracle.grad(x)
    analytic_hess = oracle.hess(x)

    fd_grad = grad_finite_diff(oracle.func, x)
    fd_hess = hess_finite_diff(oracle.func, x)

    print("Тестовая точка x:")
    print(x)
    print("\nАналитический градиент:")
    print(analytic_grad)
    print("\nЧисленное приближение градиента:")
    print(fd_grad)
    print("Норма разности градиентов:", np.linalg.norm(analytic_grad - fd_grad))

    print("\nАналитический гессиан:")
    print(analytic_hess)
    print("\nЧисленное приближение гессиана:")
    print(fd_hess)
    print("Норма разности гессианов:", np.linalg.norm(analytic_hess - fd_hess))
    print("\n" + "="*70 + "\n")

print("#" * 5 + " ПРОВЕРКА ДЛЯ ПЛОТНОЙ МАТРИЦЫ " + "#" * 20 + "\n")
oracle_dense = create_log_reg_oracle(A_dense, b, regcoef, oracle_type='usual')
for idx, x in enumerate(test_points):
    print(f"Проверка для тестовой точки {idx + 1}:")
    check_grad_hess(oracle_dense, x)

print("#" * 5 + " ПРОВЕРКА ДЛЯ РАЗРЯЖЕННОЙ МАТРИЦЫ " + "#" * 20 + "\n")
oracle_sparse = create_log_reg_oracle(A_sparse, b, regcoef, oracle_type='usual')
for idx, x in enumerate(test_points):
    print(f"Проверка для тестовой точки {idx + 1}:")
    check_grad_hess(oracle_sparse, x)
