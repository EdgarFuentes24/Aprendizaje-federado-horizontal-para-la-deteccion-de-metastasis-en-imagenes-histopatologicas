import torch
from torch.utils.data import DataLoader, Subset


def build_feature_extractor(resnet_fn, weights, device: torch.device) -> torch.nn.Module:
    """
    Carga una ResNet preentrenada, congela sus pesos y elimina la capa FC final,
    devolviendo un extractor de características listo para inferencia.

    Parameters:
    resnet_fn : callable   – función constructora (resnet18, resnet34, …)
    weights   : WeightsEnum – pesos preentrenados (p.ej. ResNet18_Weights.DEFAULT)
    device    : torch.device

    Returns:
    extractor : torch.nn.Sequential  – produce tensores [B, D]"""

    model = resnet_fn(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    # Elimina la capa de clasificación (última capa FC)
    extractor = torch.nn.Sequential(*list(model.children())[:-1])
    extractor.to(device)
    extractor.eval()

    return extractor


def extract_features(
    dataset,
    transform,
    extractor: torch.nn.Module,
    hospital_ids: list,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    """
    Extrae features para los hospitales indicados dentro de un split.

    Returns:
    results : dict  { hospital_id: {"features": Tensor, "labels": Tensor, "slides": Tensor} }"""
    subset = dataset.get_subset(split, transform=transform)
    hospital_ids_array = subset.metadata_array[:, 0]  # columna 0 = hospital

    results = {}

    for hospital_id in hospital_ids:

        indices = torch.where(hospital_ids_array == hospital_id)[0]
        hospital_subset = Subset(subset, indices)

        loader = DataLoader(
            hospital_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        features_list, labels_list, slides_list = [], [], []

        with torch.no_grad():
            for images, labels, batch_metadata in loader:

                images = images.to(device, non_blocking=True)
                features = extractor(images)
                features = features.view(features.size(0), -1)  # [B, D]

                features_list.append(features.cpu())
                labels_list.append(labels)
                slides_list.append(batch_metadata[:, 1])  # columna 1 = slide

        results[hospital_id] = {
            "features": torch.cat(features_list),
            "labels":   torch.cat(labels_list),
            "slides":   torch.cat(slides_list),
        }

    return results