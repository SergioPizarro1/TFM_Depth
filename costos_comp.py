import pandas as pd

modelo = "DPT"  # "BTS" o "DPT"
if modelo == "BTS":
    df = pd.read_csv("metricas_bts.csv")
elif modelo == "DPT":
    df = pd.read_csv("metricas_dpt.csv")

# Buscar el último índice donde epoch == 0
last_start = df[df["epoch"] == 0].index.max()

# Quedarse con el último entrenamiento
df = df.loc[last_start:].reset_index(drop=True)

# Calcular métricas
tiempo_medio_epoca = df["epoch_time_sec"].mean()
tiempo_total = df["epoch_time_sec"].sum()
gpu_media = df["gpu_peak_mb"].mean()

print(f"=== Métricas de costos de entrenamiento {modelo} ===")
print("Tiempo medio por época: {:.2f} segundos".format(tiempo_medio_epoca))
print("Tiempo total de entrenamiento: {:.2f} segundos ({:.2f} horas)".format(
    tiempo_total, tiempo_total/3600))
print("Consumo medio de GPU: {:.1f} MB ({:.2f} GB)".format(
    gpu_media, gpu_media/1024))