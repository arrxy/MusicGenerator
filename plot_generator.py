import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

CSV_FILE = "optimal_results.csv"
PLOT_FILE = "scaling_plot.png"

def plot_from_csv():
    if not os.path.exists(CSV_FILE):
        print(f"{CSV_FILE} not found.")
        return
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print("\n--- Loaded Data ---")
    print(df)
    
    if len(df) < 2:
        print("\nNot enough data")
        return

    x_data = df['params'].values
    y_data = df['val_loss'].values
    model_names = df['model'].values

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, s=150, c='red', zorder=5, label='Trained Models')
    for i, txt in enumerate(model_names):
        plt.annotate(txt, (x_data[i], y_data[i]), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)

    # Power Law Fit: L(N) = a * N^(-alpha) + c
    if len(df) >= 3:
        try:
            def power_law(N, a, alpha, c):
                return a * np.power(N, -alpha) + c
            p0 = [10, 0.1, 0.5] 
            popt, _ = curve_fit(power_law, x_data, y_data, p0=p0, maxfev=10000)
            a_fit, alpha_fit, c_fit = popt
            print(f"\nScaling Law Fit: Alpha = {alpha_fit:.4f}")
            x_range = np.linspace(min(x_data) * 0.8, max(x_data) * 1.2, 100)
            plt.plot(x_range, power_law(x_range, *popt), 'b--', label=f'Power Law (α={alpha_fit:.2f})')
        except:
            print("\nCould not fit perfect power law ")
            plt.plot(x_data, y_data, 'b--', alpha=0.5, label='Trend')
    else:
        plt.plot(x_data, y_data, 'b--', alpha=0.5, label='Trend')

    plt.xscale('log')
    plt.xlabel('Parameters (N)', fontsize=12)
    plt.ylabel('Validation Loss (L)', fontsize=12)
    plt.title('Transformer Scaling Laws (Symbolic Music)', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    plt.savefig(PLOT_FILE)
    print(f"\nPlot saved to {PLOT_FILE}")

if __name__ == "__main__":
    plot_from_csv()