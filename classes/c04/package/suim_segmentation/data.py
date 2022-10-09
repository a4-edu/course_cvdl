from pathlib import Path
import random
from typing import Tuple, List


import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils import data as tdata


class SuimDataset(tdata.Dataset):
    IMAGES_FOLDER = "images"
    MASKS_FOLDER = "masks"

    LABEL_COLORS = (
        ("Background(waterbody)", "000"),
        ("Human divers", "001"),
        ("Aquatic plants and sea-grass", "010"),
        ("Wrecks and ruins", "011"),
        ("Robots (AUVs/ROVs/instruments)", "100"),
        ("Reefs and invertebrates", "101"),
        ("Fish and vertebrates", "110"),
        ("Sea-floor and rocks", "111")
    )

    def __init__(self, root: Path, masks_as_color: bool = True, target_size: Tuple[int, int]=None):
        self.root = Path(root)
        self.samples = self._load_dataset(self.root, masks_as_color=masks_as_color)
        self.target_size = target_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает пару тензоров (изображение и маску) по индексу сэмпла.
        Тензоры должны быть приведены к размеру target_size, если target_size не None
        (например, через transforms.functional.resize).
        """
        assert (0 <= idx < len(self))
        img, mask = self.samples[idx]
        if self.target_size is None:
            return (img, mask)
        img = transforms.functional.resize(img, self.target_size)
        mask = transforms.functional.resize(mask, self.target_size, interpolation=transforms.functional.InterpolationMode.NEAREST)
        return (img, mask)

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"<SuimDataset({len(self)})|{self.root}>"

    @classmethod
    def _load_dataset(cls, root: Path, masks_as_color: bool) -> List[ Tuple[torch.Tensor, torch.Tensor] ]:
        if not root.exists():
            raise ValueError(f"Root {root.absolute()} does not exist")
        images_root = root / cls.IMAGES_FOLDER
        masks_root = root / cls.MASKS_FOLDER

        assert images_root.exists(), f"Images path {images_root.absolute()} does not exist"
        assert masks_root.exists(), f"Masks path {masks_root.absolute()} does not exist"

        images_paths = list(images_root.glob("*.jpg"))
        masks_paths = []
        for img_path in images_paths:
            mask_path = masks_root / f"{img_path.stem}.bmp"
            assert mask_path.exists(), f"{mask_path.absolute()} does not exist"
            masks_paths.append(mask_path)

        samples = []
        to_tensor = transforms.PILToTensor()
        for (img_path, mask_path) in zip(images_paths, masks_paths):
            img = to_tensor(Image.open(img_path)).float() / 255
            mask = to_tensor(Image.open(mask_path))
            if not masks_as_color:
                mask = cls._mask_color_to_label(mask)
            samples.append([img, mask])
        return samples

    @classmethod
    def _mask_color_to_label(cls, mask_color: torch.Tensor) -> torch.Tensor:
        """
        Конвертирует маску[3, H, W] из цвета (0-0-0, 0-0-255 ... 255-255-255) в маску с индексом класса [1, H, W]
        """
        assert mask_color.dtype == torch.uint8, mask_color.dtype
        mask_color01 = (mask_color > 0).byte()
        mask_label = mask_color01[0] + 2 * mask_color01[1] + 4 * mask_color01[2]
        return mask_label[None]


class EveryNthFilterSampler(tdata.Sampler):
    def __init__(self, *, dataset_size: int, n: int, pass_every_nth: bool, shuffle: bool):
        self.dataset_size = dataset_size
        self.n = n
        self.pass_every_nth = pass_every_nth
        self.shuffle = shuffle

    def _get_index_list(self) -> List[int]:
        """
        Возвращает список индексов датасета, из которого потом сэмплирует DataLoader.
        Если pass_every_nth (пропускать каждый n-ый),
            то возвращает все индексы, которые нацело делятся на n,
            иначе возвращает все индексы, которые НЕ делятся нацело на n
        """
        if (self.pass_every_nth):
            return self.n * np.arange( self.dataset_size // self.n )
        else:
            x = np.arange(self.dataset_size)
            return x[x % self.n != 0]

    def __len__(self):
        return len(self._get_index_list())

    def __iter__(self):
        indices = self._get_index_list()
        if self.shuffle:
            random.shuffle(indices)
        return (idx for idx in indices)
