import os
from dataclasses import dataclass

import numpy as np
from matplotlib.pyplot import imread


@dataclass
class Sample:
    name: str
    image: np.ndarray
    label: np.ndarray
    mask: np.ndarray


def load_dataset() -> tuple[list[Sample], list[Sample]]:
    """Charge les images des ensembles d'entrainement et de test dans 2 listes d'objets de classe Sample. Pour chaque échantillon, il faut créer un objet de classe Sample contenant les propriétés "name", "image", "label" et "mask". N'oubliez pas de stocker l'objet sample à chaque itération de la boucle for. 
    
    On pourra accéder à la première image du dataset d'entrainement avec :
        train[0].image .
    """
    files = sorted(os.listdir("../DRIVE/data/training/"))
    train = []

    for file in files:
        image = imread(os.path.join("../DRIVE/data/training/", file))
        label = imread(os.path.join("../DRIVE/label/training/", file))
        mask = imread(os.path.join("../DRIVE/mask/training/", file))

        sample = Sample(name=file, image=image, label=label, mask=mask)
        train.append(sample)

    files = sorted(os.listdir("../DRIVE/data/test/"))
    test = []

    for file in files:
        image = imread(os.path.join("../DRIVE/data/test/", file))
        label = imread(os.path.join("../DRIVE/label/test/", file))
        mask = imread(os.path.join("../DRIVE/mask/test/", file))

        sample = Sample(name=file, image=image, label=label, mask=mask)
        test.append(sample)

    return train, test
