import numpy as np
from sklearn.preprocessing import StandardScaler

from src.fedHEONN import FedHEONN_classifier, FedHEONN_coordinator
from src.data_utils import prepare_eval_data, to_onehot, compute_metrics

# Entrenamiento federado

def train_federated(hospitals_data, lam, encrypted=False):
    """Entrena un modelo FedHEONN mediante aprendizaje federado.

    En el aprendizaje federado cada hospital actúa como un cliente
    independiente: los datos nunca abandonan su nodo local. Cada cliente
    normaliza sus propias características, entrena su modelo local y
    envía al coordinador únicamente los parámetros intermedios M y US
    El coordinador agrega estos parámetros y calcula los pesos globales óptimos.

    La normalización es siempre local porque, en un escenario federado
    real, cada nodo desconoce las estadísticas de los demás hospitales.

    Parameters:    
    hospitals_data : list of dict
        Lista con los datos de cada hospital cliente.
        Cada elemento es un dict con claves "features" y "labels".
    lam : float
        Parámetro de regularización L2 del coordinador FedHEONN.

    Returns:
    coordinator : FedHEONN_coordinator  con pesos globales agregados"""

    coordinator = FedHEONN_coordinator(f="logs", lam=lam, encrypted=encrypted)

    M_list  = []
    US_list = []

    for data in hospitals_data:
        X = data["features"].cpu().numpy()
        y = data["labels"].cpu().numpy()

        # Normalización local
        scaler = StandardScaler().fit(X)
        X      = scaler.transform(X)

        # Entrenamiento del cliente local
        client = FedHEONN_classifier(f="logs", encrypted=encrypted)
        client.fit(X.T, to_onehot(y))

        # Solo se envían los parámetros
        M_list.append(client.M)
        US_list.append(client.US)

    # Agregación en el coordinador
    coordinator.aggregate(M_list, US_list)

    return coordinator


# Evaluación

def evaluate_federated(coordinator, data, encrypted=False):
    """Evalúa el modelo federado sobre un conjunto de datos.

    Los pesos globales del coordinador se distribuyen a un cliente
    receptor, que realiza la inferencia sobre los datos normalizados
    localmente.

    Parameters:
    coordinator : FedHEONN_coordinator  con pesos agregados
    data        : dict  con claves "features" y "labels"

    Returns:
    metrics : dict  con acc, auc, f1, sensitivity, specificity (en %)"""

    X, y = prepare_eval_data(data, normalization="local")

    client = FedHEONN_classifier(f="logs", encrypted=encrypted)
    client.set_weights(coordinator.send_weights())

    y_pred = client.predict(X)

    y_prob = client._predict(X)[1, :]

    return compute_metrics(y, y_pred, y_prob)