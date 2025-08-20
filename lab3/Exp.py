import numpy as np
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from collections import defaultdict
import os
from optimization import barrier_method_lasso, newton
from oracles import LASSOOptOracle, lasso_duality_gap 

save_path = r"C:\Users\Redmi\Desktop\Методы оптимизации\лаб3\графики2"
np.random.seed(42)
def run_experiment(A, b, reg_coef, x0, u0, params):
    result = barrier_method_lasso(
        A, b, reg_coef, x0, u0,
        tolerance=params.get('tolerance', 1e-5),
        tolerance_inner=params.get('tolerance_inner', 1e-8),
        max_iter=params.get('max_iter', 100),
        max_iter_inner=params.get('max_iter_inner', 20),
        t_0=params.get('t_0', 1),
        gamma=params.get('gamma', 10),
        c1=params.get('c1', 1e-4),
        lasso_duality_gap=lasso_duality_gap,
        trace=True,
        display=False
    )
    return result[2]  

def plot_results(histories, labels, title):
    plt.figure(figsize=(15, 6))

    plt.subplot(121)
    for history, label in zip(histories, labels):
        if history and 'duality_gap' in history:
            gaps = history['duality_gap']
            plt.plot(range(len(gaps)), gaps, label=label)
    plt.yscale('log')

    plt.xlabel('Число итераций')
    plt.ylabel('Зазор двойственности (log)')
    plt.title(f'{title} (по итерациям)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(122)
    for history, label in zip(histories, labels):
        if history and 'duality_gap' in history and 'time' in history:
            gaps = history['duality_gap']
            times = history['time']
            plt.semilogy(times, gaps, label=label)
    plt.xlabel('Реальное время работы (сек)')
    plt.yscale('log')
    plt.ylabel('Зазор двойственности (log)')
    plt.title(f'{title} (по времени)')
    plt.legend()
    plt.grid(True)
    filename = title.replace(' ', '_').replace('(', '').replace(')', '') + '.png'
    plt.tight_layout()
    full_path = os.path.join(save_path, filename)
    
    plt.savefig(full_path)
    print(f"График сохранён в {full_path}")

def experiment_sensitivity():
    
    n, m  = 500, 1000
    reg_coef = 1/m
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    x0 = np.zeros(n)
    u0 = np.ones(n)
    
    gamma_values = [1, 5, 10, 50, 100]
    histories_gamma = []
    for gamma in gamma_values:
        params = {'gamma': gamma}
        history = run_experiment(A, b, reg_coef, x0, u0, params)
        histories_gamma.append(history)
    
    plot_results(histories_gamma, [f'γ={g}' for g in gamma_values], 
                'Чувствительность к выбору параметра γ')
    
    # Исследование tolerance_inner
    tol_values = [1e-2, 1e-4, 1e-6, 1e-8]
    histories_tol = []
    for tol in tol_values:
        params = {'tolerance_inner': tol}
        history = run_experiment(A, b, reg_coef, x0, u0, params)
        histories_tol.append(history)
    
    plot_results(histories_tol, [f'ε={t}' for t in tol_values], 
                'Чувствительность к выбору параметра ε')

def experiment_dimensions():

    
    base_m, base_n = 1000, 500
    reg_coef = 1/base_m

    # Исследование размерности n
    n_values = [10, 50, 100, 200, 500]
    histories_n = []

    for n in n_values:
        m = base_m
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x0 = np.random.uniform(0, 1, size=n)
        u0 = np.ones(n)

        history = run_experiment(A, b, reg_coef, x0, u0, {})
        histories_n.append(history)
    
    plot_results(histories_n, [f'n={n}' for n in n_values], 
                'Поведение метода для различных значений раз-ти про-ва n')
    

    m_values = [20, 100, 200, 500, 1000]
    histories_m = []
    for m in m_values:
        reg_coef = 1/m
        n = min(m//2, base_n) 
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x0 = np.random.uniform(0, 1, size=n)
        u0 = np.ones(n)
        history = run_experiment(A, b, reg_coef, x0, u0, {})
        histories_m.append(history)
    
    plot_results(histories_m, [f'm={m}' for m in m_values], 
                'Поведение метода для различных значений размера выборки m')


    lambda_values = [0.001, 0.01, 0.1, 1, 10, ]
    histories_lambda = []
    for lam in lambda_values:
        n, m = base_n, base_m
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x0 = np.random.uniform(0, 1, size=n)
        u0 = np.ones(n)
        history = run_experiment(A, b, lam, x0, u0, {})
        histories_lambda.append(history)
    
    plot_results(histories_lambda, [f'λ={lam}' for lam in lambda_values], 
                'Поведение метода для различных значений коэффициента λ ')




if __name__ == "__main__":
    experiment_sensitivity() 
    experiment_dimensions()  