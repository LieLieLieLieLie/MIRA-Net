from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import h5py
except Exception as exc:  # pragma: no cover
    h5py = None
    H5PY_IMPORT_ERROR = exc
else:
    H5PY_IMPORT_ERROR = None


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class HEp2Dataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], image_size: int, augment: bool, noise_range: tuple[int, int]):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.noise_range = noise_range

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.rotate(random.uniform(-25, 25), resample=Image.BILINEAR)
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        return img

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float().div(255.0)
        if img.mode == "L":
            return data.view(img.height, img.width).unsqueeze(0)
        channels = len(img.getbands())
        return data.view(img.height, img.width, channels).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        if self.augment:
            img = self._augment(img)
        clean = self._to_tensor(img)
        sigma = random.uniform(*self.noise_range) / 255.0
        noisy = (clean + torch.randn_like(clean) * sigma).clamp(0, 1)
        return (noisy - 0.5) / 0.5, (clean - 0.5) / 0.5, torch.tensor(label, dtype=torch.long)


class CamelyonH5Dataset(Dataset):
    def __init__(self, indices: np.ndarray, x_path: str | Path, labels: np.ndarray, image_size: int, augment: bool):
        self.indices = np.asarray(indices)
        self.x_path = str(x_path)
        self.labels = labels
        self.image_size = image_size
        self.augment = augment
        self._h5 = None

    def __len__(self) -> int:
        return len(self.indices)

    def _file(self):
        if h5py is None:
            raise RuntimeError(f"h5py cannot be imported: {H5PY_IMPORT_ERROR}")
        if self._h5 is None:
            self._h5 = h5py.File(self.x_path, "r")
        return self._h5

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.rotate(random.choice([0, 90, 180, 270]), resample=Image.BILINEAR)
        if random.random() < 0.4:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
            img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.15))
        return img

    @staticmethod
    def _to_normalized_tensor(img: Image.Image) -> torch.Tensor:
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float().div(255.0)
        tensor = data.view(img.height, img.width, 3).permute(2, 0, 1).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])
        f = self._file()
        key = "x" if "x" in f else list(f.keys())[0]
        arr = f[key][real_idx]
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        if self.augment:
            img = self._augment(img)
        return self._to_normalized_tensor(img), torch.tensor(int(self.labels[real_idx]), dtype=torch.long)

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def scan_hep2(root: str | Path) -> list[tuple[str, int]]:
    root = Path(root)
    samples: list[tuple[str, int]] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        try:
            label = int(folder.name) - 1
        except ValueError:
            continue
        samples.extend((str(p), label) for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    return samples


def _stratified_cap(samples: list[tuple[str, int]], max_samples: int, seed: int) -> list[tuple[str, int]]:
    if max_samples <= 0 or max_samples >= len(samples):
        return samples
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[str, int]]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)
    selected: list[tuple[str, int]] = []
    remaining = max_samples
    classes = sorted(by_class)
    for i, cls in enumerate(classes):
        pool = by_class[cls][:]
        rng.shuffle(pool)
        quota = remaining // (len(classes) - i)
        take = min(len(pool), quota)
        selected.extend(pool[:take])
        remaining -= take
    rng.shuffle(selected)
    return selected


def split_samples(samples: list[tuple[str, int]], val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    train: list[tuple[str, int]] = []
    val: list[tuple[str, int]] = []
    test: list[tuple[str, int]] = []
    by_class: dict[int, list[tuple[str, int]]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)
    for cls_samples in by_class.values():
        cls_samples = cls_samples[:]
        rng.shuffle(cls_samples)
        n = len(cls_samples)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        test.extend(cls_samples[:n_test])
        val.extend(cls_samples[n_test:n_test + n_val])
        train.extend(cls_samples[n_test + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_hep2_loaders(cfg: dict[str, Any]):
    task = cfg["hep2"]
    samples = scan_hep2(cfg["data"]["hep2_dir"])
    if not samples:
        raise FileNotFoundError(f"No HEp-2 images found under {cfg['data']['hep2_dir']}")
    samples = _stratified_cap(samples, int(task["max_samples"]), cfg["project"]["seed"])
    train, val, test = split_samples(samples, cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], cfg["project"]["seed"])

    train_labels = np.array([label for _, label in train])
    class_count = np.bincount(train_labels, minlength=task["classes"])
    weights = 1.0 / np.maximum(class_count[train_labels], 1)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(train), replacement=True)

    noise_range = (task["noise_sigma_min"], task["noise_sigma_max"])
    kwargs = {"num_workers": cfg["data"]["num_workers"], "pin_memory": True}
    train_ds = HEp2Dataset(train, task["image_size"], True, noise_range)
    val_ds = HEp2Dataset(val, task["image_size"], False, noise_range)
    test_ds = HEp2Dataset(test, task["image_size"], False, noise_range)
    return {
        "train": DataLoader(train_ds, batch_size=task["batch_size"], sampler=sampler, **kwargs),
        "val": DataLoader(val_ds, batch_size=task["batch_size"], shuffle=False, **kwargs),
        "test": DataLoader(test_ds, batch_size=task["batch_size"], shuffle=False, **kwargs),
        "sizes": {"train": len(train), "val": len(val), "test": len(test), "total_used": len(samples)},
    }


def load_camelyon_labels(y_path: str | Path, n_total: int) -> np.ndarray:
    y_path = Path(y_path)
    if not y_path.exists():
        raise FileNotFoundError(
            "Camelyon labels are missing. Put camelyonpatch_level_2_split_train_y.h5 "
            f"next to the x file, or set data.camelyon_y. Missing path: {y_path}"
        )
    if h5py is None:
        raise RuntimeError(f"h5py cannot be imported: {H5PY_IMPORT_ERROR}")
    with h5py.File(y_path, "r") as f:
        key = "y" if "y" in f else list(f.keys())[0]
        labels = np.asarray(f[key]).reshape(-1).astype(int)
    if len(labels) < n_total:
        raise ValueError(f"Camelyon label count {len(labels)} is smaller than image count {n_total}")
    return labels


def _h5_len(x_path: str | Path) -> int:
    if h5py is None:
        raise RuntimeError(f"h5py cannot be imported: {H5PY_IMPORT_ERROR}")
    with h5py.File(x_path, "r") as f:
        key = "x" if "x" in f else list(f.keys())[0]
        return int(f[key].shape[0])


def _cap_indices(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    indices = np.arange(len(labels))
    if max_samples <= 0 or max_samples >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    parts = []
    classes = np.unique(labels)
    remaining = max_samples
    for i, cls in enumerate(classes):
        cls_idx = indices[labels == cls].copy()
        rng.shuffle(cls_idx)
        quota = remaining // (len(classes) - i)
        take = min(len(cls_idx), quota)
        parts.append(cls_idx[:take])
        remaining -= take
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def build_camelyon_loaders(cfg: dict[str, Any]):
    if h5py is None:
        raise RuntimeError(
            "h5py is unavailable in this Python environment. Reinstall dependencies with: "
            "pip install -r requirements.txt"
        )
    task = cfg["camelyon"]
    seed = cfg["project"]["seed"]
    train_x = Path(cfg["data"]["camelyon_x"])
    train_y = Path(cfg["data"]["camelyon_y"])
    val_x = Path(cfg["data"].get("camelyon_valid_x", ""))
    val_y = Path(cfg["data"].get("camelyon_valid_y", ""))
    test_x = Path(cfg["data"].get("camelyon_test_x", ""))
    test_y = Path(cfg["data"].get("camelyon_test_y", ""))
    if not train_x.exists():
        raise FileNotFoundError(f"Camelyon x file not found: {train_x}")
    n_train = _h5_len(train_x)
    train_labels = load_camelyon_labels(train_y, n_train)[:n_train]
    train = _cap_indices(train_labels, int(task["max_samples"]), seed)

    if val_x.exists() and val_y.exists() and test_x.exists() and test_y.exists():
        n_val = _h5_len(val_x)
        n_test = _h5_len(test_x)
        val_labels = load_camelyon_labels(val_y, n_val)[:n_val]
        test_labels = load_camelyon_labels(test_y, n_test)[:n_test]
        val = _cap_indices(val_labels, int(task.get("max_val_samples", 0)), seed + 1)
        test = _cap_indices(test_labels, int(task.get("max_test_samples", 0)), seed + 2)
    else:
        labels = train_labels
        indices = train
        rng = np.random.default_rng(seed)
        train_parts = []
        val_parts = []
        test_parts = []
        for cls in np.unique(labels[indices]):
            cls_idx = indices[labels[indices] == cls].copy()
            rng.shuffle(cls_idx)
            n = len(cls_idx)
            n_test = max(1, round(n * cfg["data"]["test_ratio"]))
            n_val = max(1, round(n * cfg["data"]["val_ratio"]))
            test_parts.append(cls_idx[:n_test])
            val_parts.append(cls_idx[n_test:n_test + n_val])
            train_parts.append(cls_idx[n_test + n_val:])
        train = np.concatenate(train_parts)
        val = np.concatenate(val_parts)
        test = np.concatenate(test_parts)
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)
        val_x = test_x = train_x
        val_labels = test_labels = train_labels

    kwargs = {"num_workers": cfg["data"]["num_workers"], "pin_memory": True}
    train_ds = CamelyonH5Dataset(train, train_x, train_labels, task["image_size"], True)
    val_ds = CamelyonH5Dataset(val, val_x, val_labels, task["image_size"], False)
    test_ds = CamelyonH5Dataset(test, test_x, test_labels, task["image_size"], False)
    return {
        "train": DataLoader(train_ds, batch_size=task["batch_size"], shuffle=True, **kwargs),
        "val": DataLoader(val_ds, batch_size=task["batch_size"], shuffle=False, **kwargs),
        "test": DataLoader(test_ds, batch_size=task["batch_size"], shuffle=False, **kwargs),
        "sizes": {"train": len(train), "val": len(val), "test": len(test), "train_total": n_train},
    }
