# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Dict, Sequence

import torch
from mmengine.dist import all_reduce_dict
from mmengine.runner.amp import autocast
from mmengine.runner.loops import ValLoop

from mmseg.registry import LOOPS


@LOOPS.register_module()
class ValLoopWithLoss(ValLoop):
    """Validation loop that reports metrics and validation loss.

    The loss pass is executed with ``torch.no_grad()`` while the model remains
    in eval mode, so it does not update weights, gradients, or BatchNorm stats.
    """

    def run(self) -> dict:
        self._loss_sums: Dict[str, float] = defaultdict(float)
        self._loss_count = 0

        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        metrics.update(self._get_loss_metrics())
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

        self._accumulate_loss(data_batch)

    def _accumulate_loss(self, data_batch: Sequence[dict]) -> None:
        model = self.runner.model
        module = model.module if hasattr(model, 'module') else model
        batch_size = len(data_batch['data_samples'])

        with torch.no_grad(), autocast(enabled=self.fp16):
            data = module.data_preprocessor(data_batch, True)
            losses = module._run_forward(data, mode='loss')
            _, log_vars = module.parse_losses(losses)

        for key, value in log_vars.items():
            if 'loss' not in key:
                continue
            self._loss_sums[key] += float(value.detach().cpu()) * batch_size
        self._loss_count += batch_size

    def _get_loss_metrics(self) -> dict:
        if self._loss_count == 0:
            return {}

        model = self.runner.model
        module = model.module if hasattr(model, 'module') else model
        device = next(module.parameters()).device
        loss_tensors = {
            key: torch.tensor(value, dtype=torch.float64, device=device)
            for key, value in self._loss_sums.items()
        }
        loss_tensors['_loss_count'] = torch.tensor(
            self._loss_count, dtype=torch.float64, device=device)
        all_reduce_dict(loss_tensors, op='sum')

        count = loss_tensors.pop('_loss_count').item()
        return {
            key: round((value / count).item(), 6)
            for key, value in loss_tensors.items()
        }
