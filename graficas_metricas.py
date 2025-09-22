# graficas_bts_tripleta.py
import pandas as pd
import matplotlib.pyplot as plt

# 1) Cargar CSV
modelo = "DPT"  # "BTS" o "DPT"
if modelo == "BTS":
    df = pd.read_csv("metricas_bts.csv")
elif modelo == "DPT":
    df = pd.read_csv("metricas_dpt.csv")

# Buscar el último índice donde epoch == 0
last_start = df[df["epoch"] == 0].index.max()

# Quedarse con el último entrenamiento
df = df.loc[last_start:].reset_index(drop=True)

# 2) Localizar la mejor época según delta1
best_idx = df["val_delta1"].idxmax()
best_epoch = int(df.loc[best_idx, "epoch"]) if "epoch" in df.columns else int(best_idx)

# 3) Crear figura con 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# --- Subplot 1: pérdidas
axs[0].plot(df["epoch"], df["train_silog"], "--", label="Pérdida train")
axs[0].plot(df["epoch"], df["val_silog"], label="Pérdida validación")
axs[0].axvline(best_epoch, color="k", linestyle=":", linewidth=1)
axs[0].axvspan(best_epoch, df["epoch"].max(), alpha=0.12)
axs[0].set_title("Pérdida SILog")
axs[0].set_xlabel("Época"); axs[0].set_ylabel("Valor")
axs[0].grid(True); axs[0].legend()

# --- Subplot 2: métricas de error
for col in ["val_absrel", "val_rmse", "val_rmse_log"]:
    axs[1].plot(df["epoch"], df[col], label=col)
axs[1].axvline(best_epoch, color="k", linestyle=":", linewidth=1)
axs[1].axvspan(best_epoch, df["epoch"].max(), alpha=0.12)
axs[1].set_title("Métricas de error")
axs[1].set_xlabel("Época")
axs[1].grid(True); axs[1].legend()

# --- Subplot 3: δ-accuracy
for col in ["val_delta1", "val_delta2", "val_delta3"]:
    axs[2].plot(df["epoch"], df[col], label=col)
axs[2].axvline(best_epoch, color="k", linestyle=":", linewidth=1)
axs[2].axvspan(best_epoch, df["epoch"].max(), alpha=0.12)
axs[2].set_title("δ-accuracy")
axs[2].set_xlabel("Época")
axs[2].grid(True); axs[2].legend()

if modelo == "BTS":
    plt.suptitle("Evolución de métricas BTS", fontsize=14)
    plt.tight_layout()
    plt.savefig("metricas_bts.png", dpi=150)
elif modelo == "DPT":
    plt.suptitle("Evolución de métricas DPT", fontsize=14)
    plt.tight_layout()
    plt.savefig("metricas_dpt.png", dpi=150)
plt.show()

print(f"Mejor δ1 en la época: {best_epoch}")

