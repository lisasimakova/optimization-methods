import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numpy.linalg import norm
from sklearn.datasets import load_svmlight_file
import optimization
import oracles

sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

out_f = "exp3"
os.makedirs(out_f, exist_ok=True)

dat_fil = [
    r"C:\Users\Redmi\Desktop\Методы оптимизации\lab2\w8a.t",
    r"C:\Users\Redmi\Desktop\Методы оптимизации\lab2\rcv1_train.binary.bz2",
    r"C:\Users\Redmi\Desktop\Методы оптимизации\lab2\gisette_scale.bz2",
    r"C:\Users\Redmi\Desktop\Методы оптимизации\lab2\real-sim.bz2",
    r"C:\Users\Redmi\Desktop\Методы оптимизации\lab2\news20.binary.bz2"
]

methods = [
    ("Gradient Descent", optimization.gradient_descent, palette[0]),
    ("Hessian-Free Newton", optimization.hessian_free_newton, palette[1]),
    ("L-BFGS", optimization.lbfgs, palette[2])
]

for dataset_path in dat_fil:
    dataset_name = os.path.basename(dataset_path)
    X, y = load_svmlight_file(dataset_path)
    lambda_reg = 1 / len(y)
    oracle = oracles.create_log_reg_oracle(X, y, lambda_reg, oracle_type='optimized')

    results = {}

    for m_name, m_func, color in methods:
        x0 = np.zeros(X.shape[1])
        _, _, history = m_func(oracle, x0, trace=True)
        results[m_name] = {"log": history, "color": color}

    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        fvals = data["log"]["func"]
        plt.plot(range(len(fvals)), fvals, label=name, color=data["color"])
    plt.xlabel("Номер итерации")
    plt.ylabel("Значение функции")
    plt.title(f"Зависимость значения функции против номера итерации метода:\n{dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_f, f"func_vs_iter_{dataset_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data["log"]["time"], data["log"]["func"], label=name, color=data["color"])
    plt.xlabel('Реальное время работы (сек)')
    plt.ylabel("Значение функции")
    plt.title(f"Зависимость значения функции против реального времени работы:\n{dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_f, f"func_vs_time_{dataset_name}.png"))
    plt.close()

    grad_init_sq = norm(oracle.grad(np.zeros(X.shape[1]))) ** 2
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        grad_vals = np.array(data["log"]["grad_norm"])
        times = np.array(data["log"]["time"])
        log_grad_ratio = ((grad_vals ** 2) / grad_init_sq)
        plt.plot(times, log_grad_ratio, label=name, color=data["color"])
    plt.xlabel('Реальное время работы (сек)')
    plt.ylabel("Относительный квадрат нормы градиента (в лог. шкале)")
    plt.yscale('log')
    plt.title(f"Зависимость относительного квадрата нормы градиента (в лог. шкале) против реального времени работы:\n{dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_f, f"grad_vs_time_{dataset_name}.png"))
    plt.close()
