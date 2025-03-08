from collections import defaultdict
from typing import Optional

import torch
from torch import Tensor
from torchmetrics.functional.classification.auroc import _auroc_compute
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TORCH_LOWER_1_6


class AggregateWindowAUROC(Metric):
    r"""Compute Area Under the Receiver Operating Characteristic Curve for a 3D
    Medical Image by aggregating window level results.

    Since CT scans vary in their number of slices, and are often too long to be
    processed at once, we split each CT scan into windows (stacks) of N number
    of slices. We then train models with these windows as inputs. During inference,
    we want to compute not only the window level performance, but also the
    series-level performance to understand the clinical implications of our model.
    Therefore, we use this class to aggregate windows-level predictions to
    study-level for computing AUROC.

    Args:
        agg (str): method for aggregating window-level prediction to study-level.
            Should be one of ('max', 'mean').
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translated to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range ``[0, num_classes-1]``
        average:
            - ``'micro'`` computes metric globally. Only works for multilabel problems
            - ``'macro'`` computes metric for each class and uniformly averages them
            - ``'weighted'`` computes metric for each class and does a weighted-average,
              where each class is weighted by their support (accounts for class imbalance)
            - ``None`` computes and returns the metric per class
        max_fpr:
            If not ``None``, calculates standardized partial AUC over the
            range ``[0, max_fpr]``. Should be a float between 0 and 1.
        num_classes (int): integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems

    Adapted from: https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/classification/auroc.py
    """

    def __init__(
        self,
        agg: Optional[str] = "max",
        pos_label: Optional[int] = None,
        average: Optional[str] = "macro",
        max_fpr: Optional[float] = None,
        num_classes: Optional[int] = 1,
    ) -> None:
        super().__init__()

        self.agg = agg
        self.num_classes = num_classes
        self.pos_label = pos_label
        self.average = average
        self.max_fpr = max_fpr

        self.results = defaultdict(list)
        self.targets = defaultdict(list)

        self.mode: DataType = None  # type: ignore

        allowed_aggregation = ("max", "mean")
        if self.agg not in allowed_aggregation:
            raise ValueError(f"Argument `agg` expected to be one of the following: {allowed_aggregation} but got {self.agg}")

        allowed_average = (None, "macro", "weighted", "micro")
        if self.average not in allowed_average:
            raise ValueError(
                f"Argument `average` expected to be one of the following: {allowed_average} but got {self.average}"
            )

        if self.max_fpr is not None:
            if not isinstance(max_fpr, float) or not 0 < max_fpr <= 1:
                raise ValueError(f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}")

            if _TORCH_LOWER_1_6:
                raise RuntimeError(
                    "`max_fpr` argument requires `torch.bucketize` which is not available below PyTorch version 1.6"
                )

    def update(self, preds: Tensor, target: Tensor, series: str) -> None:  # type: ignore
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
            series: name of series for which the window belongs to
        """

        preds = preds.detach()
        target = target.detach()

        for p, t, s in zip(preds, target, series):
            self.results[s].append(p)
            self.targets[s].append(t)

    def compute(self) -> torch.Tensor:

        # flatten results
        preds = []
        targets = []
        for k in self.results.keys():
            series_pred = torch.stack(self.results[k])

            if self.agg == "max":
                preds.append(series_pred.max(axis=0)[0])
            elif self.agg == "mean":
                preds.append(series_pred.mean(axis=0))
            else:
                raise Exception("Aggregation method should be either max or mean")

            series_target = torch.stack(self.targets[k]).max(axis=0)[0]
            targets.append(series_target)

        targets = torch.stack(targets)
        preds = torch.stack(preds)
        probs = torch.sigmoid(preds)

        aurocs = []
        if self.num_classes == 1:
            p = probs.unsqueeze(1)
            t = targets.unsqueeze(1)
            auroc = _auroc_compute(
                p,
                t,
                self.mode,
                self.num_classes,
                self.pos_label,
                self.average,
                self.max_fpr,
            )
            aurocs = [auroc]
        else:
            for cls_idx in range(probs.shape[-1]):
                p = probs[:, cls_idx].unsqueeze(1)
                t = targets[:, cls_idx].unsqueeze(1)
                auroc = _auroc_compute(
                    p,
                    t,
                    self.mode,
                    self.num_classes,
                    self.pos_label,
                    self.average,
                    self.max_fpr,
                )
                aurocs.append(auroc)
        return aurocs

    def reset(self):
        # reset
        self.results = defaultdict(list)
        self.targets = defaultdict(list)
