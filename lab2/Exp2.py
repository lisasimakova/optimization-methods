import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.datasets import load_svmlight_file
import optimization
import oracles

def graf_first():
    plt.figure(figsize=(10, 6))
    for mu, col in zip(m_s, сolor_):
        g_v = np.array(results[mu]['grad_norm'])
        l_r = ((g_v ** 2) / init_grad_sq)
        plt.plot(range(len(l_r)), l_r, label=f"l={mu}", col=col)
    plt.xlabel("Номер итерации", fontsize=12)
    plt.ylabel("Относительный квадрат нормы градиента (в лог. шкале)" , fontsize=12)
    plt.yscale('log')
    plt.title("Зависимость относительного квадрата нормы градиента от номера итерации", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'log_grad_norm_vs_iter.png'))

def graf_second():
    plt.figure(figsize=(10, 6))
    for mu, col in zip(m_s, сolor_):
        g_v = np.array(results[mu]['grad_norm'])
        timestamps = np.array(results[mu]['time'])
        l_r = ((g_v ** 2) / init_grad_sq)
        plt.plot(timestamps, l_r, label=f"l={mu}", col=col)
    plt.xlabel('Реальное время работы (сек)', fontsize=12)
    plt.ylabel("Относительный квадрат нормы градиента (в лог. шкале)", fontsize=12)
    plt.yscale('log')
    plt.title("Зависимость относительного квадрата нормы градиента от реального времени работы", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'log_grad_norm_vs_time.png'))

dataset_path = r"C:\Users\Redmi\Desktop\Методы оптимизации\lab2\gisette_scale.bz2" 
output_dir = r"C:\Users\Redmi\Desktop\Методы оптимизации\lab2\Эксперимент 2.2"

features, labels = load_svmlight_file(dataset_path)
reg_coef = 1 / len(labels)
oracle = oracles.create_log_reg_oracle(features, labels, reg_coef, oracle_type='optimized')

m_s = [0, 1, 5, 10, 50, 100, 500, 1000]
сolor_ = ['tab:gray', 'tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']

results = {}

for m in m_s:
    solution, info, log = optimization.lbfgs(oracle, np.zeros(features.shape[1]), trace=True, muory_size=m)
    results[m] = log

init_grad_sq = np.linalg.norm(oracle.grad(np.zeros(features.shape[1]))) ** 2

os.makedirs(output_dir, exist_ok=True)
graf_first()
graf_second()



