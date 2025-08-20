import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelBinarizer
from datetime import datetime
import matplotlib.cm as cm
from oracles import create_log_reg_oracle
import optimization
from sklearn.utils import shuffle

d_f = [
    r"C:\Users\Redmi\Desktop\Методы оптимизации\lab2\w8a.t",
    r"C:\Users\Redmi\Desktop\Методы оптимизации\lab2\rcv1_train.binary.bz2",
    r"C:\Users\Redmi\Desktop\Методы оптимизации\lab2\real-sim.bz2",
    r"C:\Users\Redmi\Desktop\Методы оптимизации\lab2\news20.binary.bz2"
]

epsilons = [1, 1e-1, 1e-2, 1e-3, 1e-4,1e-5, 1e-6,1e-7, 1e-8]

save_dir = r"C:\Users\Redmi\Desktop\Методы оптимизации\lab2\exp6"
os.makedirs(save_dir, exist_ok=True)

log_file_path = os.path.join(save_dir, 'logs.txt')
log_file = open(log_file_path, 'w', encoding='utf-8')

e_r = {path: [] for path in d_f}

for path in d_f:
    log_file.write(f"\n### Датасет: {os.path.basename(path)} ###\n")
    X, y = load_svmlight_file(path)
    X, y = shuffle(X, y, random_state=42)

    lb = LabelBinarizer(pos_label=1, neg_label=-1)
    # y = lb.fit_transform(y).ravel()
    m, n = X.shape

    split = int(0.8 * m)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    oracle = create_log_reg_oracle(
        A=X_train,
        b=y_train,
        regcoef = 1 / X_train.shape[0],
        oracle_type='usual'
    )
    x0 = np.zeros(n)

    for eps in epsilons:
        start = datetime.now()
        x_star, msg, hist = optimization.lbfgs(
            oracle,
            x0,
            tolerance=eps
        )
        duration = (datetime.now() - start).total_seconds()
        logits = X_test.dot(x_star)
        y_pred = np.where(logits >= 0, 1, -1)
        error = np.mean(y_pred != y_test)

        e_r[path].append(error * 100)

        log_file.write(f"Точность ε={eps:<9.1e} | Ошибка классификации={error:.4f} | Время={duration:.2f} сек\n")

log_file.close()

plt.figure(figsize=(10, 6))

colors = plt.get_cmap('Set2').colors
for idx, (path, errors) in enumerate(e_r.items()):
    label = os.path.basename(path)
    color = colors[idx % len(colors)]
    plt.plot(epsilons, errors, marker='o', label=label, color=color)

plt.xscale('log')
plt.xlabel('Точность оптимизации ε', fontsize=12)
plt.ylabel('Процент ошибки классификации', fontsize=12)
plt.title('Зависимость ошибки классификации от точности оптимизации', fontsize=13)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

fig_path = os.path.join(save_dir, 'error_vs_tolerance.png')
plt.savefig(fig_path, dpi=300)
plt.show()
