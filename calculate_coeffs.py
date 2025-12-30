import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def analyze_profiling(filename="vllm_profiling_data.csv"):
    df = pd.read_csv(filename)
    
    print("--- Analysis Report ---")

    # 1. Calculate Beta_1 (Prefill Cost)
    # Filter for prefill steps
    prefill_df = df[df['type'] == 'prefill']
    X_pre = prefill_df[['num_prefill_tokens']]
    y_pre = prefill_df['duration_ms']
    
    reg_pre = LinearRegression()
    reg_pre.fit(X_pre, y_pre)
    
    beta_1 = reg_pre.coef_[0] * 1000 # Convert ms to microseconds for your Go code
    print(f"Beta_1 (Compute/Token): {beta_1:.4f} µs")

    # 2. Calculate Beta_0, Beta_2, Beta_3 (Decode Costs)
    # Filter for decode steps
    decode_df = df[df['type'] == 'decode']
    
    # We fit: Time = Beta_0 + (Beta_2 * BatchSize) + (Beta_3 * TotalKV)
    X_dec = decode_df[['num_decode_reqs', 'total_kv_usage']]
    y_dec = decode_df['duration_ms']
    
    reg_dec = LinearRegression()
    reg_dec.fit(X_dec, y_dec)
    
    beta_0 = reg_dec.intercept_ * 1000       # Base overhead (µs)
    beta_2 = reg_dec.coef_[0] * 1000         # Per Request overhead (µs)
    beta_3 = reg_dec.coef_[1] * 1000         # Per KV Token overhead (µs)
    
    print(f"Beta_0 (Base Overhead): {beta_0:.4f} µs")
    print(f"Beta_2 (Per Req Head):  {beta_2:.4f} µs")
    print(f"Beta_3 (Per KV Token):  {beta_3:.4f} µs")
    
    # 3. Calculate 'Best Loss' (MSE of the Decode fit)
    preds = reg_dec.predict(X_dec)
    mse = np.mean((y_dec - preds) ** 2)
    print(f"Best Loss (MSE):        {mse:.4f}")

    print("\n--- Proposed Config for Simulation ---")
    print("alpha_coeffs:")
    print("  - [Use Defaults or Profile Tokenizer Separately]")
    print("beta_coeffs:")
    print(f"  - {beta_0:.4f}  # Base")
    print(f"  - {beta_1:.4f}  # Prefill")
    print(f"  - {beta_2:.4f}  # Decode Request")
    print(f"  - {beta_3:.4f}  # Decode KV Memory (NEW)")

if __name__ == "__main__":
    analyze_profiling()