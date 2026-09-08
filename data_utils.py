import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    confusion_matrix,
)


# Normalización

def fit_global_scaler(hospitals_data):
    """Ajusta un StandardScaler sobre los datos de todos los hospitales.

    En el modo centralizado es posible calcular estadísticas globales
    porque todos los datos están disponibles en un único nodo. Se
    concatenan las características de todos los hospitales antes de
    ajustar el scaler para que la normalización sea consistente.

    Parameters:
    hospitals_data : list of dict
        Lista de diccionarios con clave "features" (torch.Tensor).

    Returns:
    scaler : StandardScaler
        Scaler ajustado sobre el conjunto completo."""
    all_X = [data["features"].cpu().numpy() for data in hospitals_data]
    return StandardScaler().fit(np.vstack(all_X))


def get_scaler(hospitals_data, normalization):
    """Devuelve el scaler adecuado según el tipo de normalización.

    Parameters:
    hospitals_data : list of dict
    normalization  : {"global", "local"}

    Returns:
    scaler : StandardScaler or None
        StandardScaler global si normalization="global"; None si "local"."""
    
    if normalization == "global":
        return fit_global_scaler(hospitals_data)
    return None


def normalize(X, normalization, scaler_global=None):
    """Normaliza un array de características.

    - "global": aplica el scaler ya ajustado sobre todos los hospitales.
    - "local" : ajusta y aplica un scaler propio sobre X.

    Parameters:
    X             : np.ndarray, shape (n_samples, n_features)
    normalization : {"global", "local"}
    scaler_global : StandardScaler or None

    Returns:
    X_norm : np.ndarray, shape (n_samples, n_features)
    scaler : StandardScaler  (el que se ha aplicado)"""
    if normalization == "global":
        return scaler_global.transform(X), scaler_global

    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


# Preparación de datos para evaluación

def prepare_eval_data(data, normalization, scaler_global=None):
    """Normaliza y transpone los datos de validación o test.

    FedHEONN espera las entradas con forma (n_features, n_samples),
    de ahí la transposición final.

    Parameters:
    data          : dict  con claves "features" (Tensor) y "labels" (Tensor)
    normalization : {"global", "local"}
    scaler_global : StandardScaler or None

    Returns:
    X : np.ndarray, shape (n_features, n_samples)
    y : np.ndarray, shape (n_samples,)"""
    X = data["features"].cpu().numpy()
    y = data["labels"].cpu().numpy()

    X, _ = normalize(X, normalization, scaler_global)
    return X.T, y


def to_onehot(y):
    """Convierte un vector de etiquetas enteras a codificación one-hot.

    Parameters:
    y : np.ndarray, shape (n_samples,)

    Returns:
    t_onehot : np.ndarray, shape (n_samples, n_classes)"""

    n_classes = len(np.unique(y))
    t_onehot  = np.zeros((len(y), n_classes))

    for i, label in enumerate(y):
        t_onehot[i, label] = 1

    return t_onehot


# Métricas de evaluación

def compute_metrics(y_true, y_pred, y_prob=None):
    """Calcula un conjunto completo de métricas de clasificación binaria.

    Además del accuracy, se incluyen métricas clínicas relevantes para
    detección de cáncer: AUC-ROC, sensibilidad, especificidad y F1-score.
    En diagnóstico médico la sensibilidad (recall de la clase positiva) es
    especialmente importante, ya que un falso negativo —tumor no detectado—
    tiene consecuencias más graves que un falso positivo.

    Parameters:
    y_true : np.ndarray, shape (n_samples,)
        Etiquetas reales.
    y_pred : np.ndarray, shape (n_samples,)
        Etiquetas predichas por el modelo.
    y_prob : np.ndarray, shape (n_samples,) or None
        Probabilidad de la clase positiva (necesaria para AUC-ROC).
        Si es None, el AUC-ROC no se calcula.

    Returns:
    metrics : dict con las siguientes claves:
        - acc         : accuracy (%)
        - auc         : AUC-ROC (%)
        - f1          : F1-score (%)
        - sensitivity : sensibilidad / recall clase positiva (%)
        - specificity : especificidad / recall clase negativa (%)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = 100.0 * tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "acc"         : 100.0 * accuracy_score(y_true, y_pred),
        "auc"         : 100.0 * roc_auc_score(y_true, y_prob) if y_prob is not None else None,
        "f1"          : 100.0 * f1_score(y_true, y_pred),
        "sensitivity" : sensitivity,
        "specificity" : specificity,
    }

    return metrics