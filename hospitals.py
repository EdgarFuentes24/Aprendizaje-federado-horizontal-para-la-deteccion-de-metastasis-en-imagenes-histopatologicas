from collections import defaultdict

import torch
from torch.utils.data import Subset


# Estadísticas del dataset por hospital

def count_samples_per_hospital(dataset):
    """Cuenta el número de muestras de cada hospital en cada split.

    Parameters:
    dataset : WILDSDataset

    Returns:
    counts : dict  { hospital_id: { split: n_samples } }"""

    splits = ["train", "val", "test"]
    counts = defaultdict(dict)

    for split in splits:
        subset   = dataset.get_subset(split)
        metadata = subset.metadata_array[:, 0]  # columna 0 = hospital

        for hospital in torch.unique(metadata):
            n = torch.sum(metadata == hospital).item()
            counts[int(hospital)][split] = n

    return counts


def print_hospital_statistics(counts):
    """Imprime por pantalla el número de imágenes por hospital y split.

    Parameters:
    counts : dict  devuelto por count_samples_per_hospital"""
    print("Número de imágenes por hospital y split:")
    print(f"{'Hospital':<12} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 40)

    for hospital in sorted(counts.keys()):
        train = counts[hospital].get("train", 0)
        val   = counts[hospital].get("val",   0)
        test  = counts[hospital].get("test",  0)
        print(f"  {hospital:<10} {train:>8,} {val:>8,} {test:>8,}")


# Filtrado por hospital

def get_hospital_subset(dataset_subset, hospital_id):
    """Devuelve un Subset con solo las muestras del hospital indicado.

    Parameters:
    dataset_subset : WILDSSubset
    hospital_id    : int

    Returns:
    subset : torch.utils.data.Subset"""
    indices = torch.where(
        dataset_subset.metadata_array[:, 0] == hospital_id
    )[0]
    return Subset(dataset_subset, indices)