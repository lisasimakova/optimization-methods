import numpy as np
import matplotlib.pyplot as plt
import sys
from oracles import QuadraticOracle
from optimization import gradient_descent
from plot_trajectory_2d import plot_levels, plot_trajectory

matrices = {
    'A1_diag_1_1': np.array([[1.0, 0.0], [0.0, 1.0]]),
    'A2_diag_100_1': np.array([[100.0, 0.0], [0.0, 1.0]]),
    'A3_non_diag': np.array([[4.0, 3.0], [3.0, 4.0]])
}

initial_points = {
    'start1': np.array([-2.0, 2.0]),
    'start2': np.array([2.0, -2.0]),
    'start3': np.array([0.0, 0.0])
}

step_const = 0.02
step_strategies = {
    'const_0.02': {'method': 'Constant', 'c': step_const},
    'armijo': {'method': 'Armijo', 'c1': 1e-4, 'alpha_0': 1.0},
    'wolfe': {'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9, 'alpha_0': 1.0}
}

strategy_names_rus = {
    'const_0.02': 'Постоянный шаг (0.02)',
    'armijo': 'Армихо',
    'wolfe': 'Вульф'
}

start_names_rus = {
    'start1': '(-2, 2)',
    'start2': '(2, -2)',
    'start3': '(0, 0)'
}

max_iter = 30

log_file = open("log.txt", "w")
original_stdout = sys.stdout
sys.stdout = log_file

for matrix_name, A in matrices.items():
    oracle = QuadraticOracle(A, b=np.array([1.0, 1.0]))
    condition_number = np.linalg.cond(A)

    for strategy_name, options in step_strategies.items():
        for start_name, x0 in initial_points.items():
            x_star, message, history = gradient_descent(
                oracle, x0, max_iter=max_iter,
                line_search_options=options, trace=True
            )

            print(f"=== Матрица: {matrix_name}, стратегия шага: {strategy_name}, Начальная точка: {start_names_rus[start_name]} ===")
            print(f"Оптимальная точка: {x_star}")
            print(f"Количество итераций: {len(history['x']) - 1}")
            print(f"Число обусловленности: {condition_number:.2f}")
            print(f"Сообщение: {message}\n")

            plt.figure(figsize=(6, 5))
            plot_levels(oracle.func, xrange=[-3.0, 3.0], yrange=[-3.0, 3.0])

            label = (
                f"Матрица: {matrix_name}, стратегия шага: {strategy_name}, начальная точка: {start_names_rus[start_name]}"
            )
            plot_trajectory(oracle.func, history['x'], label=label)

            plt.title(f"Траектория градиентного спуска ({matrix_name})", fontsize=11, fontweight='bold')
            plt.xlabel(r"$\mathbf{X}$", fontsize=10, fontweight='bold')
            plt.ylabel(r"$\mathbf{Y}$", rotation=0, fontsize=10, fontweight='bold')
            plt.legend(loc='best', fontsize=8)
            plt.tight_layout()
            filename = f"{matrix_name}_{strategy_name}_{start_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

sys.stdout = original_stdout
log_file.close()
