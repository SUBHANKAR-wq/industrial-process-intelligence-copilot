import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/autoencoder_output_keras.csv"
OUTPUT_IMAGE = "data/processed/reconstruction_error_plot.png"

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp")

plt.figure(figsize=(14, 6))
plt.plot(df["timestamp"], df["reconstruction_error"])
plt.yscale("log")  
plt.xlabel("Time")
plt.ylabel("Reconstruction Error (log scale)")
plt.title("Reconstruction Error vs Time (Log Scale)")
plt.grid(True)
plt.tight_layout()




plt.savefig(OUTPUT_IMAGE, dpi=150)
plt.close()

print(f"Plot saved to {OUTPUT_IMAGE}")
