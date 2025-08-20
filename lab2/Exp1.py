import numpy as np 
import matplotlib.pyplot as plt
import sys
import scipy.sparse as sp
from oracles import QuadraticOracle
from optimization import conjugate_gradients


np.random.seed(42)  
seeds = np.random.randint(1, 10000, 20)

n_values = [10, 100, 1000, 10000]
kappa_values = np.logspace(0, 3, num=20)
max_iter = 2500

colors = {10: 'red', 100: 'blue', 1000: 'green', 10000: "brown"}

log_file = open("log_experiment_family.txt", "w")
original_stdout = sys.stdout
sys.stdout = log_file

plt.figure(figsize=(10, 9))

for n in n_values:
    all_curves = []

    for seed in seeds:
        np.random.seed(seed)
        b = np.random.randn(n)
        iteration_counts = []

        for kappa in kappa_values:
            r = np.random.rand(n)
            a = r * (kappa - 1) + 1
            if n >= 2:
                a[0] = 1.0
                a[1] = kappa

            A = sp.diags(a)
            oracle = QuadraticOracle(A, b=b)
            x0 = np.zeros(n)

            x_star, message, history = x_star, message, history = conjugate_gradients(lambda x: A @ x, b, x0, max_iter=max_iter, trace=True)
            iters = len(history['time']) - 1
            iteration_counts.append(iters)

            print(f"seed = {seed}, n = {n}, kappa = {kappa:.2e}, iterations = {iters}, message = {message}")

        all_curves.append(iteration_counts)
        plt.plot(kappa_values, iteration_counts, linestyle=':', color=colors[n], alpha=0.4)

    mean_curve = np.mean(np.array(all_curves), axis=0)
    plt.plot(kappa_values, mean_curve, linestyle='-', color=colors[n], label=f"n = {n}")

sys.stdout = original_stdout
log_file.close()

plt.xscale('log')
plt.xlabel("Число обусловленности κ", fontsize=12)
plt.ylabel("Число итераций T", fontsize=12)
plt.title("Семейства кривых: число итераций метода сопряженных градиентов\nв зависимости от κ и размерности", fontsize=14)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()

plt.savefig("iterations_vs_condition_family.png", dpi=300, bbox_inches="tight")
plt.show()
