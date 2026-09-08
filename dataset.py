from wilds import get_dataset
from torchvision import transforms

from src.config import DATA_DIR


def get_camelyon_dataset(root_dir: str = DATA_DIR):
    """
    Descarga (si hace falta) y devuelve el dataset Camelyon17 de WILDS
    junto con el transform estándar de ImageNet.

    Returns:
    dataset   : WILDSDataset
    transform : torchvision.transforms.Compose"""

    dataset = get_dataset(
        dataset="camelyon17",
        root_dir=root_dir,
        download=False,
    )

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return dataset, transform