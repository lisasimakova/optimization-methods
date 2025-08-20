import numpy as np
from matplotlib import pyplot as plt
import optimization as optimization
import oracles as oracles
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

def compare(svm_file, save_to_file):
    A, b = load_svmlight_file(svm_file) 
    num_nonzero = A.nnz
    print(f'Количество ненулевых значений: {num_nonzero}')
    
    coef = 1.0 / b.size
    oracle = oracles.create_log_reg_oracle(A, b, coef)

    x0 = np.zeros(A.shape[1])
    
    _, _, hs_fd = optimization.gradient_descent(oracle, x0, trace=True)
    elapsed_gd = hs_fd['time']
    
    plt.figure(figsize=(10, 9))
    plt.xlabel('Время, сек')
    plt.ylabel('Значение функции')
    plt.plot(elapsed_gd, hs_fd['func'], label="Градиентный спуск")

    _, _, hs_n = optimization.newton(oracle, x0, trace=True)
    elapsed_n = hs_n['time']    
    plt.plot(elapsed_n, hs_n['func'], label="Ньютон")
    plt.legend()
    plt.title('Сходимость значения функции')
    plt.savefig(save_to_file + '_func.png', dpi=300)
    plt.clf()
    
    grad0 = oracle.grad(x0)
    
    norm_grad0_sq = np.linalg.norm(grad0)**2

    plt.figure(figsize=(10, 9))
    rel_grad_gd = np.square(np.array(hs_fd['grad_norm'])) / norm_grad0_sq
    rel_grad_n = np.square(np.array(hs_n['grad_norm'])) / norm_grad0_sq
    
    plt.plot(elapsed_gd, rel_grad_gd, label="Градиентный спуск")
    plt.plot(elapsed_n, rel_grad_n, label="Ньютон")
    plt.xlabel('Время, сек')
    plt.ylabel('Относительное значение квадрата нормы градиента')
    plt.yscale('log')
    plt.title('Сходимость нормы градиента')
    plt.legend()
    plt.savefig(save_to_file + '_grad.png', dpi=300)
    plt.clf()
    

if __name__ == "__main__":
    compare('w8a.txt', 'w8a')
    compare('gisette_scale', 'gisette_scale')
    compare('real-sim', 'real-sim')
