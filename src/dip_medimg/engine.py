from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import numpy as np
import torch
from tqdm import tqdm

from .plotting import plot_confusion_matrix, plot_history
from .utils import save_json


def accuracy_score_np(y_true: list[int], y_pred: list[int]) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return float((y_t == y_p).mean()) if len(y_t) else 0.0


def confusion_matrix_np(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def f1_from_cm(cm: np.ndarray) -> tuple[float, float, dict[str, dict[str, float]]]:
    report: dict[str, dict[str, float]] = {}
    supports = cm.sum(axis=1)
    f1s = []
    weights = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        report[str(i)] = {"precision": precision, "recall": recall, "f1-score": f1, "support": int(supports[i])}
        f1s.append(f1)
        weights.append(supports[i])
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_f1 = float(np.average(f1s, weights=weights)) if np.sum(weights) else 0.0
    return macro_f1, weighted_f1, report


def extended_metrics(cm: np.ndarray) -> dict[str, float]:
    cm = cm.astype(float)
    total = cm.sum()
    po = np.trace(cm) / total if total else 0.0
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = float(np.dot(row, col) / (total * total)) if total else 0.0
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) else 0.0

    recalls = []
    precisions = []
    specificities = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = col[i] - tp
        fn = row[i] - tp
        tn = total - tp - fp - fn
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        precisions.append(tp / (tp + fp) if tp + fp else 0.0)
        specificities.append(tn / (tn + fp) if tn + fp else 0.0)

    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = float(((tp * tn) - (fp * fn)) / denom) if denom else 0.0
    else:
        c = np.trace(cm)
        s = total
        p = col
        t = row
        denom = np.sqrt((s**2 - np.dot(p, p)) * (s**2 - np.dot(t, t)))
        mcc = float((c * s - np.dot(p, t)) / denom) if denom else 0.0

    return {
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_specificity": float(np.mean(specificities)) if specificities else 0.0,
        "mcc": mcc,
        "cohen_kappa": float(kappa),
    }


def binary_auc(y_true: list[int], scores: np.ndarray) -> float | None:
    y = np.asarray(y_true)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_ranks = ranks[:len(pos)]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def macro_ovr_auc(y_true: list[int], probs: np.ndarray, num_classes: int) -> float | None:
    aucs = []
    y = np.asarray(y_true)
    for cls in range(num_classes):
        binary = (y == cls).astype(int).tolist()
        auc = binary_auc(binary, probs[:, cls])
        if auc is not None:
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else None


def _batch_to_device(batch, device: torch.device, reconstruct: bool):
    if reconstruct:
        noisy, clean, labels = batch
        return noisy.to(device), clean.to(device), labels.to(device)
    if len(batch) == 3:
        images, _, labels = batch
        return images.to(device), None, labels.to(device)
    images, labels = batch
    return images.to(device), None, labels.to(device)


class Trainer:
    def __init__(self, model, loaders, criterion, optimizer, scheduler, device: torch.device, out_dir: str | Path, task_name: str, class_names: list[str], reconstruct: bool, checkpoint_path: str | Path | None = None):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.out_dir = Path(out_dir)
        self.task_name = task_name
        self.class_names = class_names
        self.reconstruct = reconstruct
        self.best_metric = -1.0
        self.param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        parts = self.task_name.split("_", 1)
        self.dataset_name = parts[0]
        self.model_name = parts[1] if len(parts) == 2 else self.task_name

    def fit(self, epochs: int) -> dict[str, list[float]]:
        start = time.time()
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(self.loaders["train"], train=True)
            val_loss, val_acc = self._run_epoch(self.loaders["val"], train=False)
            self.scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            print(f"[{self.task_name}] epoch {epoch:03d}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
            if val_acc > self.best_metric:
                self.best_metric = val_acc
                checkpoint = {
                    "model": self.model.state_dict(),
                    "model_state_dict": self.model.state_dict(),
                    "dataset": self.dataset_name,
                    "model_name": self.model_name,
                    "task_name": self.task_name,
                    "class_names": self.class_names,
                    "num_classes": len(self.class_names),
                    "reconstruct": self.reconstruct,
                    "parameters": self.param_count,
                    "val_acc": val_acc,
                    "epoch": epoch,
                }
                torch.save(checkpoint, self.out_dir / f"{self.task_name}_best.pt")
                torch.save(checkpoint, self.out_dir / f"{self.task_name}_checkpoint.pt")
        plot_history(history, self.out_dir, self.task_name)
        history["training_seconds"] = [time.time() - start]
        self._last_history = history   # stash for collect_run in evaluate()
        return history

    def _run_epoch(self, loader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in tqdm(loader, leave=False):
                images, clean, labels = _batch_to_device(batch, self.device, self.reconstruct)
                logits, recon = self.model(images, reconstruct=self.reconstruct)
                loss, _ = self.criterion(logits, labels, recon, clean)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()
                total_loss += float(loss.detach().cpu()) * labels.size(0)
                pred = logits.argmax(dim=1)
                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(pred.detach().cpu().tolist())
        return total_loss / max(len(loader.dataset), 1), accuracy_score_np(y_true, y_pred)

    def evaluate(self) -> dict[str, Any]:
        ckpt = self.checkpoint_path or (self.out_dir / f"{self.task_name}_best.pt")
        if ckpt.exists():
            state = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state["model"])
        else:
            print(f"[{self.task_name}] checkpoint not found: {ckpt}; evaluating current model weights")
        self.model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        y_prob: list[list[float]] = []

        # ── image sample collectors for visual panels ────────────────────
        _parts_tmp   = self.task_name.split("_", 1)
        _ds_tmp      = _parts_tmp[0]
        _is_proposed = (len(_parts_tmp) > 1 and _parts_tmp[1] == "proposed")
        _img_samples: list[dict] = []
        _MAX_IMG_SAMPLES = 120

        with torch.no_grad():
            for batch in tqdm(self.loaders["test"], leave=False):
                images, clean, labels = _batch_to_device(batch, self.device, self.reconstruct)

                if _is_proposed and self.reconstruct:
                    logits, recon = self.model(images, reconstruct=True)
                else:
                    logits, recon = self.model(images, reconstruct=False)

                probs_b = torch.softmax(logits, dim=1)
                preds_b = logits.argmax(dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds_b.cpu().tolist())
                y_prob.extend(probs_b.cpu().tolist())

                if _is_proposed and len(_img_samples) < _MAX_IMG_SAMPLES:
                    B = images.shape[0]
                    for bi in range(B):
                        if len(_img_samples) >= _MAX_IMG_SAMPLES:
                            break
                        lbl  = int(labels[bi].cpu())
                        pred = int(preds_b[bi].cpu())
                        scr  = float(probs_b[bi, pred].cpu())
                        if _ds_tmp == "hep2":
                            noisy_np = images[bi, 0].cpu().numpy().astype(np.float32)
                            clean_np = clean[bi, 0].cpu().numpy().astype(np.float32) \
                                       if clean is not None else np.zeros_like(noisy_np)
                            recon_np = recon[bi, 0].cpu().numpy().astype(np.float32) \
                                       if recon is not None else None
                            attn_np = None
                            if recon_np is not None:
                                attn_np = np.abs(recon_np - clean_np)
                                attn_np = (attn_np - attn_np.min()) / \
                                          (attn_np.max() - attn_np.min() + 1e-8)
                            _img_samples.append({
                                "noisy": noisy_np, "clean": clean_np,
                                "recon": recon_np, "attn":  attn_np,
                                "label": lbl, "pred": pred, "score": scr,
                            })
                        else:
                            mean = np.array([0.485, 0.456, 0.406], np.float32)
                            std  = np.array([0.229, 0.224, 0.225], np.float32)
                            img_np = images[bi].cpu().numpy()
                            img_np = (img_np * std[:, None, None] + mean[:, None, None])
                            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
                            _img_samples.append({
                                "img": img_np, "label": lbl,
                                "pred": pred, "score": scr,
                            })

        if _is_proposed and _img_samples:
            from .plotting import store_image_samples
            store_image_samples(_ds_tmp, _img_samples)
        cm = confusion_matrix_np(y_true, y_pred, len(self.class_names))
        macro_f1, weighted_f1, report = f1_from_cm(cm)
        metrics: dict[str, Any] = {
            "accuracy": accuracy_score_np(y_true, y_pred),
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "parameters": self.param_count,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        metrics.update(extended_metrics(cm))
        probs = np.asarray(y_prob)
        if len(self.class_names) == 2 and probs.size:
            auc = binary_auc(y_true, probs[:, 1])
            if auc is not None:
                metrics["roc_auc"] = auc
            else:
                metrics["roc_auc_note"] = "ROC AUC unavailable because one class is absent in the evaluated split."
        elif probs.size:
            auc = macro_ovr_auc(y_true, probs, len(self.class_names))
            if auc is not None:
                metrics["macro_ovr_auc"] = auc
        plot_confusion_matrix(cm, self.class_names, self.out_dir, self.task_name)
        save_json(metrics, self.out_dir / f"{self.task_name}_metrics.json")
        # ── register this run for combined visualisation ──────────────────
        from .plotting import collect_run
        _parts  = self.task_name.split("_", 1)
        _ds     = _parts[0]
        _model  = _parts[1] if len(_parts) == 2 else self.task_name
        collect_run(
            dataset=_ds,
            model=_model,
            history=getattr(self, "_last_history", {}),
            metrics=metrics,
        )
        return metrics
