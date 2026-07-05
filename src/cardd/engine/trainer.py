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


def _unwrap(model: nn.Module) -> nn.Module:
    """Devuelve el modelo real, quitando el envoltorio DataParallel si existe.

    Multi-GPU: cuando el modelo va envuelto en ``nn.DataParallel``, sus pesos y
    métodos viven en ``model.module``. Este helper centraliza el desenvoltorio para
    que el guardado/carga de checkpoints produzca claves SIN el prefijo ``module.``
    → un checkpoint entrenado en 2 GPUs se carga tal cual en 1 GPU."""
    return model.module if isinstance(model, nn.DataParallel) else model


def build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    name = str(get(cfg, "train.optimizer.name", "adamw")).lower()
    lr = float(get(cfg, "train.optimizer.lr", 1e-4))
    wd = float(get(cfg, "train.optimizer.weight_decay", 0.05))
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        mom = float(get(cfg, "train.optimizer.momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd)
    raise ValueError(f"optimizer {name!r} no soportado")


def build_scheduler(cfg, optimizer, total_steps: int):
    sched = get(cfg, "train.scheduler.name", "cosine")
    if not sched or str(sched).lower() == "none":
        return None  # lr constante (útil para el overfit smoke: loss→0 rápido)
    warmup = int(get(cfg, "train.scheduler.warmup_steps", 0))
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
    """Encapsula el loop. Se construye con ``(cfg, model, train_loader, val_loader=None)``.

    El ``train_loader`` viene de ``data.build.build_dataloader`` (D), con el
    collate que entrega ``(tuple(images), tuple(targets))`` — justo lo que espera
    ``MaskRCNN.forward``.

    Protocolo train/val/test: si se pasa ``val_loader``, cada ``train.val_interval``
    epochs se evalúa val SIN gradiente (el modelo nunca entrena sobre val) y se
    guarda ``best.pth`` con el mayor ``mask_AP`` de val. ``test`` no participa acá:
    C lo evalúa al cierre con ``best.pth`` (set no visto).

    Multi-GPU (Kaggle T4 x2): si hay >1 GPU CUDA disponible, el modelo se envuelve
    automáticamente en ``nn.DataParallel`` y cada batch se reparte entre las GPUs.
    Para desactivarlo aunque haya 2 GPUs, exporta ``CUDA_VISIBLE_DEVICES=0`` antes
    de lanzar el entrenamiento.
    """

    def __init__(self, cfg, model: nn.Module, train_loader, val_loader=None) -> None:
        self.cfg = cfg
        self.log = get_logger("cardd.engine.trainer")
        self.device = torch.device(get(cfg, "train.device", "cuda"))
        self.model = model.to(self.device)

        # --- Multi-GPU (DataParallel) -------------------------------------
        # Si hay más de una GPU CUDA visible, repartir cada batch entre ellas.
        # El modelo real sigue accesible vía _unwrap() para guardar/cargar pesos
        # sin el prefijo "module." → checkpoints compatibles con 1 GPU.
        self.n_gpus = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if self.n_gpus > 1:
            self.log.info("multi-GPU: usando %d GPUs (DataParallel)", self.n_gpus)
            self.model = nn.DataParallel(self.model)
        # ------------------------------------------------------------------

        self.loader = train_loader
        self.val_loader = val_loader

        self.epochs = int(get(cfg, "train.epochs", 1))
        self.accum = max(1, int(get(cfg, "train.grad_accum_steps", 1)))
        self.amp = bool(get(cfg, "train.amp", True)) and self.device.type == "cuda"
        self.log_interval = int(get(cfg, "train.log_interval", 10))
        self.val_interval = max(1, int(get(cfg, "train.val_interval", 1)))
        self.grad_clip = get(cfg, "train.grad_clip", None)
        self.ckpt_interval = int(get(cfg, "train.ckpt_interval", 0))  # 0 = solo last.pth al final
        self.out_dir = Path(get(cfg, "output.dir", "outputs/run"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float("-inf")

        # early stopping OPCIONAL (default OFF → schedule fijo, no afecta la ablación)
        es = "train.early_stopping"
        self.es_enabled = bool(get(cfg, f"{es}.enabled", False))
        self.es_patience = max(1, int(get(cfg, f"{es}.patience", 5)))
        self.es_min_delta = float(get(cfg, f"{es}.min_delta", 0.001))
        self.es_counter = 0

        steps_per_epoch = max(1, math.ceil(len(self.loader) / self.accum))
        self.total_steps = steps_per_epoch * self.epochs
        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(cfg, self.optimizer, self.total_steps)
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.amp)
        self.tb = SummaryWriter(str(self.out_dir / "tb"))
        self.step = 0
        self.start_epoch = 0  # se adelanta si se hace resume()

    def warm_start(self, ckpt_path) -> None:
        """Warm-start (NO estricto): carga SOLO los pesos del modelo y arranca un
        entrenamiento FRESCO (epoch 0, optimizer/scheduler nuevos del yaml actual).
        A diferencia de resume(), el lr vuelve a subir/decae segun el schedule de
        ESTA config -> sirve para extender con un schedule corto y exprimir lo que
        quedo. optimizer/scheduler/epoca/step del checkpoint se ignoran a proposito.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device)
        _unwrap(self.model).load_state_dict(ckpt["model"])
        self.log.info(
            "warm-start desde %s - pesos cargados - schedule nuevo (epoch 0/%d, lr fresco)",
            ckpt_path, self.epochs,
        )

    def resume(self, ckpt_path) -> None:
        """Resume ESTRICTO: restaura modelo + optimizer + scheduler + época + best.

        Mismo modelo (load_state_dict estricto: si el backbone difiere, falla ruidoso
        — eso es correcto, no se reanuda otra arquitectura). Pensado para Colab: si la
        sesión se corta, `--resume outputs/.../last.pth` continúa exactamente donde iba.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device)
        _unwrap(self.model).load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_metric = float(ckpt.get("best_metric", float("-inf")))
        self.step = int(ckpt.get("step", 0))
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.log.info(
            "resume desde %s · sigue en epoch %d/%d · best_metric=%.4f · step=%d",
            ckpt_path, self.start_epoch, self.epochs, self.best_metric, self.step,
        )

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
        # log de valores RESUELTOS: si una clave de base no matchea y cae a default,
        # se ve acá (no se enmascara como pasó con optim.* vs train.optimizer.*).
        self.log.info(
            "optimizer=%s · lr=%.2e · scheduler=%s",
            type(self.optimizer).__name__,
            self.optimizer.param_groups[0]["lr"],
            get(self.cfg, "train.scheduler.name", "cosine"),
        )
        last = self.out_dir / "last.pth"  # default si el loop no corre (resume ya completo)
        for epoch in range(self.start_epoch, self.epochs):
            self._run_epoch(epoch)

            # validación SIN gradiente + selección de best.pth por mask_AP de val
            if self.val_loader is not None and (epoch + 1) % self.val_interval == 0:
                val_ap = self._validate(epoch)
                improved = val_ap > self.best_metric + self.es_min_delta
                if val_ap > self.best_metric:
                    self.best_metric = val_ap
                    self._save(self.out_dir / "best.pth", epoch)
                    self.log.info("nuevo best.pth · val mask_AP=%.4f (epoch %d)", val_ap, epoch)
                # early stopping OPCIONAL (default OFF: no corta, best.pth ya queda guardado)
                if self.es_enabled:
                    self.es_counter = 0 if improved else self.es_counter + 1
                    if self.es_counter >= self.es_patience:
                        self.log.info("early stopping en epoch %d (%d vals sin mejora > %.4g)",
                                      epoch, self.es_patience, self.es_min_delta)
                        # last.pth cada época: se sobrescribe → reanudable si Colab corta
                        numbered = bool(self.ckpt_interval and (epoch + 1) % self.ckpt_interval == 0)
                        last = self.save_checkpoint(epoch, numbered=numbered)
                        break

            # last.pth cada época (estado completo): reanudable si Colab corta.
            # numbered opcional (ckpt_interval>0) si además se quiere histórico.
            numbered = bool(self.ckpt_interval and (epoch + 1) % self.ckpt_interval == 0)
            last = self.save_checkpoint(epoch, numbered=numbered)
        self.tb.close()
        return last

    def _run_epoch(self, epoch: int) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        n = len(self.loader)
        for i, (images, targets) in enumerate(self.loader):
            images, targets = self._to_device(images, targets)

            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                loss_dict = self.model(images, targets)  # train mode → dict de losses
                # Multi-GPU: DataParallel devuelve un tensor por GPU en cada loss
                # (shape [n_gpus]) en vez de un escalar. Se reduce con .mean() para
                # volver a un escalar antes de sumar. En 1 GPU, .mean() sobre un
                # escalar es no-op → seguro en ambos casos.
                loss = sum(v.mean() for v in loss_dict.values())

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
                        self.tb.add_scalar(f"train/{k}", float(v.mean().detach()), self.step)
                    self.tb.add_scalar("train/lr", lr, self.step)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """Evalúa val SIN gradiente y devuelve mask_AP. No toca los pesos.

        Usa el loop **memory-safe** (serializa a RLE por imagen, no acumula
        máscaras densas) → no hay OOM en val completa. Corre el evaluador de E
        con ``with_boundary=False`` (val más rápida; el Boundary AP es para el
        cierre, no para seleccionar best.pth) y ``image_ids`` = todas las inferidas
        (incluye las vacías como recall miss).

        Multi-GPU: la inferencia se hace con el modelo REAL (sin DataParallel), ya
        que el loop de inference de E procesa imagen a imagen y no necesita el
        reparto por batch. Se pasa _unwrap(self.model).
        """
        from cardd.engine.inference import run_inference_coco_results
        from cardd.evaluation.coco_eval import evaluate

        ds = self.val_loader.dataset
        cat_id_to_name = {
            cid: ds.label_to_name[label] for cid, label in ds.cat_id_to_label.items()
        }

        eval_model = _unwrap(self.model)
        eval_model.eval()
        coco_results, image_ids = run_inference_coco_results(
            eval_model, self.val_loader, ds, device=self.device, amp=self.amp)
        eval_model.train()

        c5 = evaluate(ds.coco, coco_results, cat_id_to_name,
                      image_ids=image_ids, with_boundary=False)

        val_ap = float(c5.get("mask_AP", float("nan")))
        box_ap = float(c5.get("box_AP", float("nan")))
        self.log.info("val · epoch %d · mask_AP=%.4f · box_AP=%.4f", epoch, val_ap, box_ap)
        self.tb.add_scalar("val/mask_AP", val_ap, self.step)
        self.tb.add_scalar("val/box_AP", box_ap, self.step)
        return val_ap

    def _save(self, path: Path, epoch: int) -> Path:
        torch.save(
            {
                # _unwrap → guarda SIN el prefijo "module." de DataParallel,
                # para que el checkpoint se cargue igual en 1 o 2 GPUs.
                "model": _unwrap(self.model).state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "epoch": epoch,
                "step": self.step,
                "best_metric": self.best_metric,
                "cfg": self.cfg,
            },
            path,
        )
        return path

    def save_checkpoint(self, epoch: int, numbered: bool = False) -> Path:
        last = self._save(self.out_dir / "last.pth", epoch)  # alias estable para evaluate/inference
        if numbered:
            self._save(self.out_dir / f"checkpoint_epoch{epoch:03d}.pth", epoch)
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
        "train": {"device": "cpu", "epochs": 1, "grad_accum_steps": 2,
                  "amp": False, "log_interval": 1,
                  "optimizer": {"name": "adamw", "lr": 1e-4},
                  "scheduler": {"name": "none"}},
        "output": {"dir": "outputs/_smoke"},
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
