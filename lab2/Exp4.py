import numpy as np
import matplotlib.pyplot as plt
import oracles, optimization
import os

memory_list = [0, 1, 5]


np.random.seed(42)
_A = np.random.randn(30, 30)
A = _A.T @ _A 
b = np.random.randn(30)
oracle = oracles.QuadraticOracle(A, b)
x0 = np.zeros(30)

l_results = {}

for l in memory_list:
    x_l, msg_l, hist_l = optimization.lbfgs(
        oracle, x0, trace=True,
        line_search_options={'method': 'Best'},
        memory_size=l,
        tolerance=1e-6,
        max_iter=300
    )

    res_l = np.array(hist_l['grad_norm'])
    l_results[l] = res_l

x_cg, msg_cg, hist_cg = optimization.conjugate_gradients(
    lambda v: A @ v, b, x0,
    trace=True,
    tolerance=1e-6,
    max_iter=300
)
res_cg = np.array(hist_cg['residual_norm'])

plt.figure(figsize=(12,10))
plt.plot(np.arange(len(res_cg)), res_cg, 'k-', label='CG')

for l, res in l_results.items():
    plt.plot(np.arange(len(res)), res, '--', label=f'L-BFGS, l={l}')

plt.xlabel("Номер итерации")
plt.title("График сходимости в терминах евклидовой нормы невязки (в лог. шкале) против номера итерации, n=30")
plt.ylabel("Евклидова норма невязки")
plt.yscale('log')
plt.legend()
plt.grid(which='major', linestyle='-', linewidth=0.7)   
plt.grid(which='minor', visible=False)    
plt.tight_layout()
os.makedirs('exp4', exist_ok=True)
plt.savefig(f'exp4/convergence_func1.png', dpi=300)
plt.close()


np.random.seed(44)
_A = np.random.randn(300, 300)
A = _A.T @ _A 
b = np.random.randn(300)
oracle = oracles.QuadraticOracle(A, b)
x0 = np.zeros(300)

l_results = {}

for l in memory_list:
    x_l, msg_l, hist_l = optimization.lbfgs(
        oracle, x0, trace=True,
        line_search_options={'method': 'Best'},
        memory_size=l,
        tolerance=1e-6,
        max_iter=1000
    )

    res_l = np.array(hist_l['grad_norm'])
    l_results[l] = res_l

x_cg, msg_cg, hist_cg = optimization.conjugate_gradients(
    lambda v: A @ v, b, x0,
    trace=True,
    tolerance=1e-6,
    max_iter=1000
)
res_cg = np.array(hist_cg['residual_norm'])


plt.figure(figsize=(12,10))
plt.plot(np.arange(len(res_cg)), res_cg, 'k-', label='CG')

for l, res in l_results.items():
    plt.plot(np.arange(len(res)), res, '--', label=f'L-BFGS, l={l}')

plt.xlabel("Номер итерации")
plt.title("График сходимости в терминах евклидовой нормы невязки (в лог. шкале) против номера итерации, n=300")
plt.ylabel("Евклидова норма невязки")
plt.yscale('log')
plt.legend()
plt.grid(which='major', linestyle='-', linewidth=0.7)  
plt.grid(which='minor', visible=False)    
plt.tight_layout()
os.makedirs('exp4', exist_ok=True)
plt.savefig(f'exp4/convergence_func2.png', dpi=300)
plt.close()