import unittest
import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_array_almost_equal, assert_almost_equal

from oracles import lasso_duality_gap
from optimization import  barrier_method_lasso


class Test(unittest.TestCase):

    def test_lasso_duality_gap(self):
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])
        regcoef = 2.0

        # Checks at point x = [0, 0, 0]
        x = np.zeros(3)
        assert_almost_equal(0.77777777777777,
                            lasso_duality_gap(x, A.dot(x) - b,
                                                      A.T.dot(A.dot(x) - b),
                                                      b, regcoef))

        # Checks at point x = [1, 1, 1]
        x = np.ones(3)
        assert_almost_equal(3.0, lasso_duality_gap(x, A.dot(x) - b,
                                                           A.T.dot(A.dot(x) - b),
                                                           b, regcoef))

    def check_prototype_results(self, results, groundtruth):
        if groundtruth[0] is not None:
            #print(np.array(results[0]))
            #print(np.array(groundtruth[0]))
            self.assertTrue(np.allclose(np.array(results[0]),
                                        np.array(groundtruth[0])))

        if groundtruth[1] is not None:
            assert_equal(results[1], groundtruth[1])

        if groundtruth[2] is not None:
            self.assertTrue(results[2] is not None)
            self.assertTrue('time' in results[2])
            self.assertTrue('func' in results[2])
            self.assertTrue('duality_gap' in results[2])
            assert_equal(len(results[2]['func']), len(groundtruth[2]))
        else:
            self.assertTrue(results[2] is None)

    def test_barrier_prototype(self):
        method = barrier_method_lasso
        A = np.eye(2)
        b = np.array([1.0, 2.0])
        reg_coef = 2.0
        x_0 = np.array([10.0, 10.0])
        u_0 = np.array([11.0, 11.0])
        ldg = lasso_duality_gap

        method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg)
        self.check_prototype_results(method(A, b, reg_coef, x_0, u_0,
                                       lasso_duality_gap=ldg, tolerance=1e10),
                                [(x_0, u_0), 'success', None])
        self.check_prototype_results(method(A, b, reg_coef, x_0, u_0,
                                       lasso_duality_gap=ldg, tolerance=1e10,
                                       trace=True),
                                [(x_0, u_0), 'success', [0.0]])
        self.check_prototype_results(method(A, b, reg_coef, x_0, u_0,
                                       lasso_duality_gap=ldg, max_iter=1,
                                       trace=True),
                                [None, 'iterations_exceeded', [0.0, 0.0]])
        method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg,
               tolerance_inner=1e-8)
        method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, max_iter=1)
        method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, max_iter_inner=1)
        method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, t_0=1)
        method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, gamma=10)
        method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, c1=1e-4)
        method(A, b, reg_coef, x_0, u_0, lasso_duality_gap=ldg, trace=True)
        method(A, b, reg_coef, x_0, u_0, 1e-5, 1e-8, 100, 20, 1, 10, 1e-4, ldg, True, True)


if __name__ == '__main__':
    unittest.main()
