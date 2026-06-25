"""Entrenador (engine).

dueño: M · fase: 4 · contrato: — (consume C1, instancia el detector)

Loop de entrenamiento bajo el presupuesto de 8 GB (RTX 5060):
  - **AMP** (precisión mixta) activado por defecto.
  - **Acumulación de gradiente**: batch físico chico (1–2) que emula uno mayor.
  - **Schedule configurable y fijo** entre filas de ablación (higiene de P).

Guarda checkpoints ``{"model","optimizer","epoch","step","cfg"}`` que
``engine.inference.build_eval_model`` sabe leer (lee la clave ``"model"``).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from cardd.utils.config import get
from cardd.utils.logging import get_logger


def build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    name = str(get(cfg, "optim.name", "adamw")).lower()
    lr = float(get(cfg, "optim.lr", 1e-4))
    wd = float(get(cfg, "optim.weight_decay", 0.05))
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        mom = float(get(cfg, "optim.momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd)
    raise ValueError(f"optimizer {name!r} no soportado")


def build_scheduler(cfg, optimizer, total_steps: int):
    sched = get(cfg, "optim.scheduler", "cosine")
    if not sched or str(sched).lower() == "none":
        return None  # lr constante (útil para el overfit smoke: loss→0 rápido)
    warmup = int(get(cfg, "optim.warmup_iters", 0))
    if str(sched).lower() == "cosine":
        parts, milestones = [], []
        if warmup > 0:
            parts.append(torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, total_iters=warmup))
            milestones.append(warmup)
        parts.append(torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup)))
        if len(parts) == 1:
            return parts[0]
        return torch.optim.lr_scheduler.SequentialLR(optimizer, parts, milestones=milestones)
    raise ValueError(f"scheduler {sched!r} no soportado")


class Trainer:
    """Encapsula el loop. Se construye con ``(cfg, model, train_loader)``.

    El ``train_loader`` viene de ``data.build.build_dataloader`` (D), con el
    collate que entrega ``(tuple(images), tuple(targets))`` — justo lo que espera
    ``MaskRCNN.forward``.
    """

    def __init__(self, cfg, model: nn.Module, train_loader) -> None:
        self.cfg = cfg
        self.log = get_logger("cardd.engine.trainer")
        self.device = torch.device(get(cfg, "train.device", "cuda"))
        self.model = model.to(self.device)
        self.loader = train_loader

        self.epochs = int(get(cfg, "train.max_epochs", 1))
        self.accum = max(1, int(get(cfg, "train.grad_accum_steps", 1)))
        self.amp = bool(get(cfg, "train.amp", True)) and self.device.type == "cuda"
        self.log_interval = int(get(cfg, "train.log_interval", 10))
        self.grad_clip = get(cfg, "train.grad_clip", None)
        self.ckpt_interval = int(get(cfg, "train.ckpt_interval", 0))  # 0 = solo last.pth al final
        self.out_dir = Path(get(cfg, "output.dir", "outputs/run"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        steps_per_epoch = max(1, math.ceil(len(self.loader) / self.accum))
        self.total_steps = steps_per_epoch * self.epochs
        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(cfg, self.optimizer, self.total_steps)
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.amp)
        self.tb = SummaryWriter(str(self.out_dir / "tb"))
        self.step = 0

    def _to_device(self, images, targets):
        images = [img.to(self.device, non_blocking=True) for img in images]
        targets = [
            {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets
        ]
        return images, targets

    def train(self) -> Path:
        self.model.train()
        self.log.info(
            "entrenando: %d epochs · ~%d opt-steps · accum=%d · amp=%s · device=%s",
            self.epochs, self.total_steps, self.accum, self.amp, self.device,
        )
        for epoch in range(self.epochs):
            self._run_epoch(epoch)
            # guardado periódico opcional (ckpt_interval=0 → solo al final): evita
            # llenar el disco y frenar el loop con un checkpoint por epoch.
            if self.ckpt_interval and (epoch + 1) % self.ckpt_interval == 0:
                self.save_checkpoint(epoch, numbered=True)
        last = self.save_checkpoint(self.epochs - 1)
        self.tb.close()
        return last

    def _run_epoch(self, epoch: int) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        n = len(self.loader)
        for i, (images, targets) in enumerate(self.loader):
            images, targets = self._to_device(images, targets)

            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                loss_dict = self.model(images, targets)  # train mode → dict de losses
                loss = sum(loss_dict.values())

            # se divide por accum para promediar el gradiente del micro-batch
            self.scaler.scale(loss / self.accum).backward()

            is_step = (i + 1) % self.accum == 0 or (i + 1) == n
            if is_step:
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.step += 1

                if self.step % self.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    loss_v = float(loss.detach())
                    self.log.info("epoch %d · step %d · loss %.4f · lr %.2e",
                                  epoch, self.step, loss_v, lr)
                    self.tb.add_scalar("train/loss", loss_v, self.step)
                    for k, v in loss_dict.items():
                        self.tb.add_scalar(f"train/{k}", float(v.detach()), self.step)
                    self.tb.add_scalar("train/lr", lr, self.step)

    def save_checkpoint(self, epoch: int, numbered: bool = False) -> Path:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": self.step,
            "cfg": self.cfg,
        }
        last = self.out_dir / "last.pth"
        torch.save(payload, last)                       # alias estable para evaluate/inference
        if numbered:
            torch.save(payload, self.out_dir / f"checkpoint_epoch{epoch:03d}.pth")
        self.log.info("checkpoint guardado (epoch %d) → %s", epoch, last)
        return last


if __name__ == "__main__":
    # Smoke del loop SIN dataset ni GPU: un "loader" sintético de 4 micro-batches C1.
    # Valida forward+backward, AMP(off en CPU), grad-accum, step y checkpoint.
    from cardd.models.detectors.mask_rcnn import build_detector

    cfg = {
        "data": {
            "num_classes": 6, "image_min_size": 320, "image_max_size": 512,
            "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225],
        },
        "model": {
            "detector": {"name": "mask_rcnn"},
            "backbone": {"name": "convnextv2_tiny", "pretrained": False},
        },
        "train": {"device": "cpu", "max_epochs": 1, "grad_accum_steps": 2,
                  "amp": False, "log_interval": 1},
        "output": {"dir": "outputs/_smoke"},
        "optim": {"name": "adamw", "lr": 1e-4, "scheduler": "none"},
    }

    def _fake_sample():
        img = torch.rand(3, 256, 256)
        tgt = {
            "boxes": torch.tensor([[10.0, 10.0, 120.0, 120.0]]),
            "labels": torch.tensor([1]),
            "masks": torch.zeros(1, 256, 256, dtype=torch.uint8),
            "image_id": 0,
        }
        return img, tgt

    fake_loader = [tuple(zip(*[_fake_sample() for _ in range(1)])) for _ in range(4)]
    model = build_detector(cfg)
    ckpt = Trainer(cfg, model, fake_loader).train()
    print("OK · checkpoint:", ckpt)