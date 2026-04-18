from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, foreground_class: int = 1) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, foreground_class]
    target_f = (target == foreground_class).float()
    inter = (probs * target_f).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
    dice = (2.0 * inter + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, ce_weight: torch.Tensor | None = None) -> torch.Tensor:
    return F.cross_entropy(logits, target, weight=ce_weight) + dice_loss(logits, target)


class SegmentationMetrics:
    def __init__(self) -> None:
        self.intersection = 0.0
        self.union = 0.0
        self.pred_sum = 0.0
        self.target_sum = 0.0
        self.correct = 0.0
        self.total = 0.0
        self.ap_scores: list[float] = []

    @staticmethod
    def average_precision(prob: torch.Tensor, target: torch.Tensor) -> float:
        scores = prob.flatten()
        labels = target.flatten().bool()
        positives = labels.sum()
        if positives == 0:
            return 0.0
        order = torch.argsort(scores, descending=True)
        labels = labels[order].float()
        tp = torch.cumsum(labels, dim=0)
        precision = tp / (torch.arange(labels.numel(), device=labels.device, dtype=torch.float32) + 1.0)
        return float((precision * labels).sum().item() / positives.item())

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probs = torch.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)
        pred_fg = pred == 1
        target_fg = target == 1

        self.intersection += float((pred_fg & target_fg).sum().item())
        self.union += float((pred_fg | target_fg).sum().item())
        self.pred_sum += float(pred_fg.sum().item())
        self.target_sum += float(target_fg.sum().item())
        self.correct += float((pred == target).sum().item())
        self.total += float(target.numel())

        for p, y in zip(probs.detach(), target_fg.detach(), strict=False):
            self.ap_scores.append(self.average_precision(p, y))

    def compute(self) -> dict[str, float]:
        iou = self.intersection / (self.union + 1e-7)
        dice = (2.0 * self.intersection) / (self.pred_sum + self.target_sum + 1e-7)
        pixel_acc = self.correct / (self.total + 1e-7)
        map_score = sum(self.ap_scores) / max(len(self.ap_scores), 1)
        return {
            "iou": iou,
            "dice": dice,
            "pixel_accuracy": pixel_acc,
            "map": map_score,
        }
