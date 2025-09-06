import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator, KroghInterpolator, lagrange

def f(x):
    return np.sin(x)

def modified_lagrange(x_data, y_data, x_eval):
    n = len(x_data)
    weights = np.ones(n)
    for j in range(n):
        for k in range(n):
            if j != k:
                weights[j] /= (x_data[j] - x_data[k])

    P = np.zeros_like(x_eval)
    for i, x in enumerate(x_eval):
        numerator = 0
        denominator = 0
        exact_node = False
        for j in range(n):
            if x == x_data[j]:
                P[i] = y_data[j]
                exact_node = True
                break
            temp = weights[j] / (x - x_data[j])
            numerator += temp * y_data[j]
            denominator += temp
        if not exact_node:
            P[i] = numerator / denominator
    return P

def run_all_interpolations(n_points):
    x_data = np.linspace(0, 1, n_points)
    y_data = f(x_data)
    x_plot = np.linspace(0, 1, 1000)
    y_true = f(x_plot)

    coeffs_std = np.polyfit(x_data, y_data, n_points - 1)
    y_std = np.polyval(coeffs_std, x_plot)

    newton_interp = KroghInterpolator(x_data, y_data)
    y_newton = newton_interp(x_plot)

    poly_lagrange = lagrange(x_data, y_data)
    y_lagrange = poly_lagrange(x_plot)

    y_modified_lagrange = modified_lagrange(x_data, y_data, x_plot)

    bary_interp = BarycentricInterpolator(x_data, y_data)
    y_bary = bary_interp(x_plot)

    plt.figure(figsize=(14, 8))
    plt.plot(x_plot, y_true, 'k--', label='sin(x)')
    plt.plot(x_plot, y_std, label='Standard Basis', alpha=0.8)
    plt.plot(x_plot, y_newton, label='Newton', alpha=0.8)
    plt.plot(x_plot, y_lagrange, label='Lagrange', alpha=0.8)
    plt.plot(x_plot, y_modified_lagrange, label='Modified Lagrange', alpha=0.8)
    plt.plot(x_plot, y_bary, label='Barycentric Interpolator', alpha=0.8)
    plt.scatter(x_data, y_data, color='black', s=10, label=f'Data points ({n_points})')
    plt.legend()
    plt.title(f'Interpolation of sin(x) with {n_points} points')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()

for n in [10, 100, 1000]:
    run_all_interpolations(n)