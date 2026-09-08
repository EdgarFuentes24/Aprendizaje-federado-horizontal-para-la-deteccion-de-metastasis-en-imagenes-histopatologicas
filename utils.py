import os
from datetime import datetime

import torch
import pandas as pd


# Gestión de directorios

def ensure_dir(path):
    """Crea el directorio indicado si no existe."""
    os.makedirs(path, exist_ok=True)


# Rutas y acceso a ficheros de características

def feature_path(features_dir, architecture, hospital_id, version):
    """Devuelve la ruta del fichero .pt para un hospital y arquitectura dados.

    Parameters:
    features_dir  : str   directorio raíz de características
    architecture  : str   nombre de la arquitectura (ej. "resnet18")
    hospital_id   : int
    version       : str   sufijo de versión (ej. "_v3")

    Returns:
    path : str
    """
    return f"{features_dir}/{architecture}_hospital{hospital_id}_features{version}.pt"


def save_features(path, features, labels, slides):
    """Guarda features, labels y slides en un fichero .pt."""
    torch.save({"features": features, "labels": labels, "slides": slides}, path)


def load_features(path):
    """Carga un fichero .pt y devuelve el diccionario con features, labels y slides."""
    return torch.load(path, map_location="cpu", weights_only=False)


# Registro de resultados experimentales

def append_results(results_dict, output_xlsx="results/all_results.xlsx"):
    """Añade una fila de resultados al Excel acumulativo de experimentos.

    Si el fichero no existe lo crea; si existe, añade la nueva fila al final.

    Parameters
    ----------
    results_dict : dict
        Métricas y metadatos del experimento (arquitectura, modo, accs, etc.).
    output_xlsx  : str
        Ruta del fichero Excel de salida.
    """
    ensure_dir(os.path.dirname(output_xlsx))

    def _to_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return x

    for key in ("val_acc", "test_acc", "time_sec", "lambda"):
        if key in results_dict:
            results_dict[key] = _to_float(results_dict[key])

    results_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_new = pd.DataFrame([results_dict])

    if os.path.exists(output_xlsx):
        df_old = pd.read_excel(output_xlsx)
        df     = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_excel(output_xlsx, index=False)
    print(f"[OK] Resultados guardados en: {output_xlsx}")