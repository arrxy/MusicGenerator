import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

CSV_FILE = "optimal_results.csv"
PLOT_FILE = "scaling_plot.png"

def plot_from_csv():
    if not os.path.exists(CSV_FILE):
        print(f"❌ File {CSV_FILE} not found. Run training first.")
        return

    # Load data
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    print("\n--- Loaded Data ---")
    print(df)
    
    if len(df) < 2:
        print("\n⚠️ Not enough data points to plot yet. Need at least 2 models trained.")
        return

    # Prepare Data
    x_data = df['params'].values
    y_data = df['val_loss'].values
    model_names = df['model'].values

    # Plot Setup
    plt.figure(figsize=(10, 6))
    
    # Scatter Plot
    plt.scatter(x_data, y_data, s=150, c='red', zorder=5, label='Trained Models')

    # Annotate points
    for i, txt in enumerate(model_names):
        plt.annotate(txt, (x_data[i], y_data[i]), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)

    # Power Law Fit: L(N) = a * N^(-alpha) + c
    if len(df) >= 3:
        try:
            def power_law(N, a, alpha, c):
                return a * np.power(N, -alpha) + c
            
            # Initial guesses usually help convergence
            p0 = [10, 0.1, 0.5] 
            popt, _ = curve_fit(power_law, x_data, y_data, p0=p0, maxfev=10000)
            
            a_fit, alpha_fit, c_fit = popt
            print(f"\n📈 Scaling Law Fit: Alpha = {alpha_fit:.4f}")
            
            # Draw curve
            x_range = np.linspace(min(x_data) * 0.8, max(x_data) * 1.2, 100)
            plt.plot(x_range, power_law(x_range, *popt), 'b--', label=f'Power Law (α={alpha_fit:.2f})')
        except:
            print("\n⚠️ Could not fit perfect power law (data might be noisy or sparse).")
            # Fallback line
            plt.plot(x_data, y_data, 'b--', alpha=0.5, label='Trend')
    else:
        # Simple connector line for < 3 points
        plt.plot(x_data, y_data, 'b--', alpha=0.5, label='Trend')

    plt.xscale('log')
    # plt.yscale('log') # Uncomment for log-log plot
    plt.xlabel('Parameters (N)', fontsize=12)
    plt.ylabel('Validation Loss (L)', fontsize=12)
    plt.title('Transformer Scaling Laws (Symbolic Music)', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    plt.savefig(PLOT_FILE)
    print(f"\n✅ Plot saved to {PLOT_FILE}")

if __name__ == "__main__":
    plot_from_csv()