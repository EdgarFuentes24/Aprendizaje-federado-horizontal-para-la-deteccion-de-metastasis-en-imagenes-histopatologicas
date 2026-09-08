"""
main.py – Extracción de características de Camelyon17 con ResNet preentrenada.

Uso:
    python main.py
"""

import torch

from src.config import (
    ARCHITECTURES,
    BATCH_SIZE,
    FEATURES_DIR,
    HOSPITAL_IDS,
    NUM_WORKERS,
    VERSION,
)
from src.dataset import get_camelyon_dataset
from src.feature_extraction import build_feature_extractor, extract_features
from src.utils import ensure_dir, feature_path, save_features


def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    ensure_dir(FEATURES_DIR)

    dataset, transform = get_camelyon_dataset()

    for arch_name, (resnet_fn, weights) in ARCHITECTURES.items():
        print(f"\n── Arquitectura: {arch_name} ──")

        extractor = build_feature_extractor(resnet_fn, weights, device)

        for split, hospital_ids in HOSPITAL_IDS.items():

            results = extract_features(
                dataset=dataset,
                transform=transform,
                extractor=extractor,
                hospital_ids=hospital_ids,
                split=split,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                device=device,
            )

            for hospital_id, data in results.items():
                path = feature_path(FEATURES_DIR, arch_name, hospital_id, VERSION)
                save_features(path, data["features"], data["labels"], data["slides"])
                print(f"  Guardado: {path}  {data['features'].shape}")


if __name__ == "__main__":
    main()