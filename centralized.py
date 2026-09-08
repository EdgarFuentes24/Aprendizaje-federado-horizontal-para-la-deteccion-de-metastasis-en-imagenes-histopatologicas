import numpy as np
import scipy as sp

from src.fedHEONN import FedHEONN_classifier
from src.data_utils import (
    get_scaler,
    normalize,
    prepare_eval_data,
    to_onehot,
    compute_metrics,
)

# Entrenamiento centralizado

def train_centralized(hospitals_data, normalization, lam):
    """Entrena un modelo FedHEONN con los datos de todos los hospitales.

    En el modo centralizado no existe privacidad de datos: todos los
    hospitales comparten sus características con un nodo central que
    concatena, mezcla y entrena un único modelo. Este modo sirve como
    cota superior de rendimiento con la que comparar el aprendizaje
    federado.

    El proceso sigue tres pasos:
      1. Normalización de características (local o global).
      2. Concatenación y aleatorización de los datos de entrenamiento.
      3. Ajuste del modelo FedHEONN y cálculo de los pesos óptimos
         mediante la descomposición SVD con regularización L2.

    Parameters:
    hospitals_data : list of dict
        Lista con los datos de cada hospital de entrenamiento.
        Cada elemento es un dict con claves "features" y "labels".
    normalization  : {"local", "global"}
        Estrategia de normalización de las características.
    lam            : float
        Parámetro de regularización L2 del modelo FedHEONN.

    Returns:
    model         : FedHEONN_classifier  con pesos ajustados
    scaler_global : StandardScaler or None"""

    # Paso 1: calcular el scaler global
    scaler_global = get_scaler(hospitals_data, normalization)

    all_X, all_y = [], []

    for data in hospitals_data:
        X = data["features"].cpu().numpy()
        y = data["labels"].cpu().numpy()

        X, _ = normalize(X, normalization, scaler_global)

        all_X.append(X)
        all_y.append(y)

    # Paso 2: concatenar y mezclar aleatoriamente
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)

    idx     = np.random.permutation(len(y_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    # Paso 3: ajustar FedHEONN y calcular pesos
    model = FedHEONN_classifier(f="logs", encrypted=False)
    model.fit(X_train.T, to_onehot(y_train))

    M_c, US_c = model.get_param()
    weights   = []

    for m, us in zip(M_c, US_c):
        U, S, _ = sp.linalg.svd(us, full_matrices=False)
        w = U @ (
            np.diag(1.0 / (S * S + lam * np.ones(len(S))))
            @ (U.T @ m)
        )
        weights.append(w)

    model.set_weights(weights)

    return model, scaler_global

# Evaluación

def evaluate_centralized(model, data, normalization, scaler_global=None):
    """Calcula métricas del modelo centralizado sobre un conjunto de datos.

    Parameters:
    model         : FedHEONN_classifier
    data          : dict  con claves "features" y "labels"
    normalization : {"local", "global"}
    scaler_global : StandardScaler or None

    Returns:
    metrics : dict  con acc, auc, f1, sensitivity, specificity (en %)"""
    X, y = prepare_eval_data(data, normalization, scaler_global)

    y_pred = model.predict(X)

    y_prob = model._predict(X)[1, :]

    return compute_metrics(y, y_pred, y_prob)