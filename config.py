from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
)

# Dispositivo de cómputo
DEVICE = "cuda"

# Rutas de directorios
FEATURES_DIR = "features"
RESULTS_DIR  = "results"
DATA_DIR     = "data"

# Partición del dataset por hospital
#
# Camelyon17-WILDS divide los 5 hospitales (centros) de la siguiente forma:
#   - Entrenamiento : hospitales 0, 3 y 4
#   - Validación    : hospital 1
#   - Test          : hospital 2
#
# Esta división es la estándar del benchmark (Koh et al., 2021) y es la que
# se respeta aquí para que los resultados sean comparables.

HOSPITAL_IDS = {
    "train": [0, 3, 4],
    "val":   [1],
    "test":  [2],
}

# Arquitecturas disponibles como extractores de características
ARCHITECTURES = {
    #"resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    #"resnet34": (resnet34, ResNet34_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
}

# Parámetros del DataLoader
BATCH_SIZE  = 32
NUM_WORKERS = 4

# Hiperparámetros de FedHEONN
LAM = 0.01
LAMBDAS = [0.001, 0.01, 0.1, 1.0]

# Versión de los ficheros generados
VERSION = "_v3"