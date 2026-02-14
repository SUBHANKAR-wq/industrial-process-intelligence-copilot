import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# CONFIG (important for Windows)

matplotlib.use("Agg")  

DATA_PATH = "data/processed/autoencoder_output_keras.csv"
OUTPUT_IMAGE = "data/processed/reconstruction_error_vs_time.png"

# LOAD DATA

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# Sort by time (CRITICAL)
df = df.sort_values("timestamp")

# Split by data source
synthetic_df = df[df["source"] == "synthetic"]
nasa_df = df[df["source"] == "nasa"]


# PLOT

fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

# ---- Synthetic Data ----
axs[0].plot(
    synthetic_df["timestamp"],
    synthetic_df["reconstruction_error"],
    linewidth=1
)
axs[0].set_title("Synthetic Data – Reconstruction Error")
axs[0].set_ylabel("Reconstruction Error")
axs[0].set_yscale("log")  
axs[0].grid(True)

# ---- NASA Data ----
axs[1].plot(
    nasa_df["timestamp"],
    nasa_df["reconstruction_error"],
    linewidth=1
)
axs[1].set_title("NASA Data – Reconstruction Error")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Reconstruction Error")
axs[1].set_yscale("log")
axs[1].grid(True)


plt.tight_layout()


# SAVE PLOT

plt.savefig(OUTPUT_IMAGE, dpi=150)
plt.close()

print("Reconstruction error plot saved successfully")
print(f"Saved at: {OUTPUT_IMAGE}")



