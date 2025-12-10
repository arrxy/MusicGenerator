import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_scaling_laws():
    # 1. Load Data
    # Assuming the files exist. If not, we create dummy data for demonstration.
    try:
        df_trans = pd.read_csv("optimal_results.csv")
        df_rnn = pd.read_csv("rnn_scaling_results.csv")
    except FileNotFoundError:
        print("❌ Data files not found. Make sure both training scripts have run.")
        return

    plt.figure(figsize=(10, 6))
    
    # 2. Plot Transformer Curve
    # Filter for the final entry of each model size if there are duplicates
    df_trans = df_trans.sort_values('params').drop_duplicates('model', keep='last')
    plt.plot(df_trans['params'], df_trans['val_loss'], 'o-', label='Transformer', linewidth=2, markersize=8)
    
    # 3. Plot RNN Curve
    df_rnn = df_rnn.sort_values('params').drop_duplicates('model', keep='last')
    plt.plot(df_rnn['params'], df_rnn['val_loss'], 's--', label='RNN (LSTM)', linewidth=2, markersize=8)

    # 4. Aesthetics (Log-Log Scale is standard for Scaling Laws)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Parameters (Log Scale)', fontsize=12)
    plt.ylabel('Validation Loss (Log Scale)', fontsize=12)
    plt.title('Scaling Law Comparison: Transformer vs. LSTM', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    
    # 5. Annotation
    plt.annotate('Transformers scale\nmore efficiently', 
                 xy=(df_trans['params'].iloc[-1], df_trans['val_loss'].iloc[-1]),
                 xytext=(df_trans['params'].iloc[-1]*0.5, df_trans['val_loss'].iloc[-1]*1.2),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig("scaling_comparison.png")
    print("💾 Plot saved to scaling_comparison.png")
    
    # 6. Efficiency Stats
    print("\n--- Efficiency Comparison ---")
    t_speed = df_trans['speed_tok_sec'].mean() if 'speed_tok_sec' in df_trans.columns else 0
    # Note: optimal_results.csv might calculate speed differently, checking headers...
    # If speed isn't in csv, we skip.
    
    if 'speed_tok_sec' in df_rnn.columns:
        r_speed = df_rnn['speed_tok_sec'].mean()
        print(f"Avg RNN Speed: {r_speed/1000:.1f}k tok/s")

if __name__ == "__main__":
    plot_scaling_laws()