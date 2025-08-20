import numpy as np
import matplotlib.pyplot as plt

from oracles import create_log_reg_oracle
from optimization import newton

step_strategies = {
    'const_0.1':       {'method': 'Constant', 'c': 0.1},
    'const_0.01':      {'method': 'Constant', 'c': 0.01},
    'Armijo_c0.1':     {'method': 'Armijo', 'c1': 0.1, 'alpha_0': 1.0},
    'Armijo_c0.5':     {'method': 'Armijo', 'c1': 0.5, 'alpha_0': 1.0},
    'Armijo_c0.9':     {'method': 'Armijo', 'c1': 0.9, 'alpha_0': 1.0},
    'Wolfe_c2_0.001':  {'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.001, 'alpha_0': 1.0},
    'Wolfe_c2_0.3':    {'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.3, 'alpha_0': 1.0},
    'Wolfe_c2_0.95':   {'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.95, 'alpha_0': 1.0},
}

strategy_names_rus = {
    'const_0.1':     'Константный шаг c=0.1',
    'const_0.01':    'Константный шаг c=0.01',
    'Armijo_c0.1':   'Бэктрэкинг c1=0.1',
    'Armijo_c0.5':   'Бэктрэкинг c1=0.5',
    'Armijo_c0.9':   'Бэктрэкинг c1=0.9',
    'Wolfe_c2_0.95': 'Вульфе c2=0.95',
    'Wolfe_c2_0.001':  'Вульфе c2=0.001',
    'Wolfe_c2_0.3': 'Вульфе c2=0.3',
}
max_iter = 50

colors = plt.cm.tab10.colors
strategies = list(step_strategies.keys())
color_map = {s: colors[i % len(colors)] for i, s in enumerate(strategies)}
color_map['Wolfe_c2_0.9'] = '#FF1493'
linestyles = {'start1': '-', 'start2': '--'}

n_samples, n_features = 500, 10
np.random.seed(42)
X = np.random.randn(n_samples, n_features)
true_w = np.random.randn(n_features)
prob = 1 / (1 + np.exp(-X.dot(true_w)))
y = 2 * (np.random.rand(n_samples) < prob).astype(float) - 1
st = 100 * np.random.rand(n_features)
initial_points_log = {
    'start1': np.ones(n_features),
    'start2': st
}
print(st, 'стартовая точка лог регрессии')
oracle_log = create_log_reg_oracle(X, y, regcoef=0.1, oracle_type='optimized')

log_iters = {name: [] for name in strategies}

fig, ax = plt.subplots(figsize=(8, 6))
for strat in strategies:
    opts = step_strategies[strat]
    for start, x0 in initial_points_log.items():
        _, msg, hist = newton(
            oracle_log, x0, max_iter=max_iter,
            line_search_options=opts, trace=True
        )
        grads = np.array(hist['grad_norm'])
        relg = grads**2 / (grads[0]**2 + 1e-16)
        ax.semilogy(relg,
                    label=f"{strategy_names_rus[strat]}, {start}",
                    color=color_map[strat],
                    linestyle=linestyles[start])
        iters = len(hist['time']) - 1
        log_iters[strat].append(iters)
        print(f"[LogReg] {start}, {strat}: {iters} it., {msg}")

ax.set_title("Метод Ньютона: стратегии выбора шага в логистической регрессии")
ax.set_xlabel("Количество итераций")
ax.set_ylabel("Относительный квадрат нормы градиента")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
print(f"Построено линий: {len(ax.lines)}")
print(f"Записано в легенду записей: {len(ax.get_legend().texts)}")
fig.savefig("conv_logreg.png", dpi=300)
plt.close(fig)

print("\nСреднее число итераций (лог-регрессия):")
for strat, its in log_iters.items():
    print(f"{strategy_names_rus[strat]}: {np.mean(its):.2f}")
    
print("\nОбщее число итераций (лог-регрессия):")
for strat, its in log_iters.items():
    print(f"{strategy_names_rus[strat]}: {its}")
