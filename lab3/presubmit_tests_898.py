import sys
import numpy as np
import pytest
import optimization
import oracles


def test_python3():
    assert sys.version_info > (3, 0)

def test_least_squares_oracle():
    A = np.eye(3)
    b = np.array([1, 2, 3])
    matvec_Ax = lambda x: A @ x
    matvec_ATx = lambda x: A.T @ x
    oracle = oracles.LeastSquaresOracle(matvec_Ax, matvec_ATx, b)

    x = np.zeros(3)
    assert pytest.approx(oracle.func(x)) == 7.0
    assert np.allclose(oracle.grad(x), [-1., -2., -3.])

    x = np.ones(3)
    assert pytest.approx(oracle.func(x)) == 2.5
    assert np.allclose(oracle.grad(x), [0., -1., -2.])

def test_least_squares_oracle_2():
    A = np.array([[1., 2.], [3., 4.]])
    b = np.array([1., -1.])
    matvec_Ax = lambda x: A @ x
    matvec_ATx = lambda x: A.T @ x
    oracle = oracles.LeastSquaresOracle(matvec_Ax, matvec_ATx, b)

    x = np.array([1., 2.])
    assert pytest.approx(oracle.func(x)) == 80.0
    assert np.allclose(oracle.grad(x), [40., 56.])

def test_l1_reg_oracle():
    oracle = oracles.L1RegOracle(1.0)

    x = np.zeros(3)
    assert pytest.approx(oracle.func(x)) == 0.0
    assert np.allclose(oracle.prox(x, alpha=1.0), x)

    x = np.array([-3.])
    assert pytest.approx(oracle.func(x)) == 3.0
    assert np.allclose(oracle.prox(x, alpha=1.0), [-2.0])
    assert np.allclose(oracle.prox(x, alpha=2.0), [-1.0])

    x = np.array([-3., 3.])
    assert pytest.approx(oracle.func(x)) == 6.0
    assert np.allclose(oracle.prox(x, alpha=1.0), [-2., 2.])
    assert np.allclose(oracle.prox(x, alpha=2.0), [-1., 1.])

def test_l1_reg_oracle_2():
    oracle = oracles.L1RegOracle(2.0)
    x = np.array([-3., 3.])
    assert pytest.approx(oracle.func(x)) == 12.0
    assert np.allclose(oracle.prox(x, alpha=1.0), [-1., 1.])



