"""CarDD · src/cardd/models/heads/pointrend.py

Dueño: Equipo C · Fase: 5 (fila a3_pointrend) · Contrato: C3 (registro HEADS)

Librería PointRend (Kirillov et al., CVPR'20) para la rama de máscara de
Mask R-CNN. Hipótesis de la fila a3: refinar bordes de scratch/crack.

Este archivo entrega los MÓDULOS (torch puro, sin tocar a M):
  - point_sample / calculate_uncertainty           (primitivas)
  - get_uncertain_point_coords_with_randomness     (muestreo en train)
  - get_uncertain_point_coords_on_grid             (subdivisión en inference)
  - StandardPointHead                              (MLP por punto, Conv1d 1x1)
  - PointRendMaskHead     -> registro HEADS "pointrend_mask_head"
  - PointRendMaskPredictor-> registro HEADS "pointrend_mask_predictor"

IMPORTANTE — alcance del hook HEADS:
El `mask_predictor` registrado es un DROP-IN válido del de torchvision: su
`forward(x)` devuelve logits COARSE con shape estándar [N, num_classes, 28, 28],
así que `maskrcnn_loss` / `maskrcnn_inference` de M siguen funcionando y la fila
a3 CORRE como máscara coarse sin tocar nada más. Pero el refinamiento por puntos
(point-loss en train, subdivisión en inference) vive en `RoIHeads`, no en el
predictor. Eso lo aplica el companion `mask_rcnn_pointrend` (próximo sub-paso),
que usa el `point_head` y las primitivas de aquí. Sin companion, a3 = coarse;
con companion, a3 = PointRend completo. Decisión asociada: gate ② (métrica de
borde) para que el refinamiento sea medible.

Las dims se leen del YAML (autodescriptivo): in_channels=256 (ancho FPN),
num_classes=7. No se hardcodea nada de M.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cardd.utils.registry import HEADS


# --------------------------------------------------------------------------- #
# Primitivas                                                                   #
# --------------------------------------------------------------------------- #
def point_sample(input: torch.Tensor, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
    """Muestrea `input` en coordenadas normalizadas con interpolación bilineal.

    Args:
        input:        [N, C, H, W].
        point_coords: [N, P, 2] o [N, P, 1, 2] en [0, 1] (x, y).
    Returns:
        [N, C, P] (si entró [N,P,2]) o [N, C, P, 1].

    grid_sample espera coords en [-1, 1]; convertimos 2*p - 1. align_corners=False
    fija la convención (mismo cuadrado que torchvision RoIAlign sampled).
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)        # [N, P, 1, 2]
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, align_corners=False, **kwargs)
    if add_dim:
        output = output.squeeze(3)                      # [N, C, P]
    return output


def calculate_uncertainty(logits: torch.Tensor, classes: torch.Tensor | None) -> torch.Tensor:
    """Incertidumbre PointRend = -|logit de la clase relevante|.

    Máxima (=0) cerca del límite de decisión (logit≈0), que es donde están los
    bordes. `logits`: [N, C, ...]; `classes`: [N] o None (si C==1, máscara
    class-agnostic). Devuelve [N, 1, ...].
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        assert classes is not None, "classes requerido si num_classes > 1"
        idx = torch.arange(logits.shape[0], device=logits.device)
        gt_class_logits = logits[idx, classes].unsqueeze(1)
    return -torch.abs(gt_class_logits)


@torch.no_grad()
def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    uncertainty_func,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    """Muestreo de puntos para el point-loss en TRAIN (Detectron2-fiel).

    Sobre-muestrea `num_points * oversample_ratio` puntos uniformes, se queda con
    los más inciertos (importance_sample_ratio) y completa con aleatorios. Devuelve
    [N, num_points, 2] en [0,1].
    """
    assert oversample_ratio >= 1
    assert 0.0 <= importance_sample_ratio <= 1.0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)          # [N, 1, num_sampled]

    num_uncertain = int(importance_sample_ratio * num_points)
    num_random = num_points - num_uncertain
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx = idx + shift[:, None]
    sampled = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain, 2)
    if num_random > 0:
        extra = torch.rand(num_boxes, num_random, 2, device=coarse_logits.device)
        sampled = torch.cat([sampled, extra], dim=1)
    return sampled


@torch.no_grad()
def get_uncertain_point_coords_on_grid(uncertainty_map: torch.Tensor, num_points: int):
    """Selección de los `num_points` más inciertos sobre una grilla (INFERENCE).

    `uncertainty_map`: [N, 1, H, W]. Devuelve (point_indices [N,P], point_coords
    [N,P,2] en [0,1] al centro de cada celda).
    """
    R, _, H, W = uncertainty_map.shape
    h_step, w_step = 1.0 / H, 1.0 / W
    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + torch.div(point_indices, W, rounding_mode="floor").to(torch.float) * h_step
    return point_indices, point_coords


# --------------------------------------------------------------------------- #
# Point head (MLP por punto)                                                   #
# --------------------------------------------------------------------------- #
class StandardPointHead(nn.Module):
    """MLP 1x1 (Conv1d) sobre [features finas ++ logits coarse] por punto.

    Entrada: fine [N, in_channels, P] y coarse [N, num_classes, P].
    Salida:  [N, num_mask_classes, P] (num_classes, o 1 si class-agnostic).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_fc: int = 3,
        fc_dim: int = 256,
        cls_agnostic_mask: bool = False,
        coarse_pred_each_layer: bool = True,
    ) -> None:
        super().__init__()
        self.coarse_pred_each_layer = coarse_pred_each_layer
        self.num_classes = num_classes
        fc_in = in_channels + num_classes        # fine ++ coarse (concat inicial)
        self.fc_layers = nn.ModuleList()
        for _ in range(num_fc):
            self.fc_layers.append(nn.Conv1d(fc_in, fc_dim, kernel_size=1))
            fc_in = fc_dim + (num_classes if coarse_pred_each_layer else 0)
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = nn.Conv1d(fc_in, num_mask_classes, kernel_size=1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        x = torch.cat((fine, coarse), dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse), dim=1)
        return self.predictor(x)


# --------------------------------------------------------------------------- #
# Componentes registrados en HEADS (slots mask_head / mask_predictor de M)     #
# --------------------------------------------------------------------------- #
@HEADS.register("pointrend_mask_head")
class PointRendMaskHead(nn.Module):
    """Slot `mask_head`: produce el mapa de features FINAS para muestrear puntos.

    Recibe las features RoI-pooled [N, in_channels, H, W] y devuelve un mapa de
    igual resolución (ligero FCN). El companion muestrea de aquí las features
    finas en las coords de los puntos inciertos.
    """

    def __init__(self, in_channels: int, conv_dim: int = 256, num_convs: int = 2) -> None:
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(num_convs):
            layers += [nn.Conv2d(ch, conv_dim, 3, padding=1), nn.ReLU(inplace=True)]
            ch = conv_dim
        self.layers = nn.Sequential(*layers)
        self.out_channels = conv_dim
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@HEADS.register("pointrend_mask_predictor")
class PointRendMaskPredictor(nn.Module):
    """Slot `mask_predictor`: DROP-IN del de torchvision + alberga el point-head.

    `forward(x)` devuelve logits COARSE [N, num_classes, 2H, 2W] (deconv stride 2),
    misma forma que `MaskRCNNPredictor` → `maskrcnn_loss`/`maskrcnn_inference` de M
    funcionan sin cambios. El refinamiento por puntos lo aplica el companion, que
    accede a `self.point_head` y a estos hiperparámetros (leídos del YAML).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dim_reduced: int = 256,
        # --- hiperparámetros PointRend (los consume el companion) ---
        num_fc: int = 3,
        fc_dim: int = 256,
        coarse_pred_each_layer: bool = True,
        cls_agnostic_mask: bool = False,
        train_num_points: int = 196,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        subdivision_steps: int = 3,
        subdivision_num_points: int = 196,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        # rama coarse: deconv 2x + conv 1x1 -> logits estándar (drop-in torchvision)
        self.conv5_mask = nn.ConvTranspose2d(in_channels, dim_reduced, 2, stride=2)
        self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1)
        # point head para el refinamiento (usado por el companion)
        self.point_head = StandardPointHead(
            in_channels=in_channels,
            num_classes=num_classes,
            num_fc=num_fc,
            fc_dim=fc_dim,
            cls_agnostic_mask=cls_agnostic_mask,
            coarse_pred_each_layer=coarse_pred_each_layer,
        )
        # config de muestreo/subdivisión, expuesta para el companion
        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.subdivision_steps = subdivision_steps
        self.subdivision_num_points = subdivision_num_points

        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits coarse, shape estándar de torchvision (drop-in)."""
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


__all__ = [
    "point_sample",
    "calculate_uncertainty",
    "get_uncertain_point_coords_with_randomness",
    "get_uncertain_point_coords_on_grid",
    "StandardPointHead",
    "PointRendMaskHead",
    "PointRendMaskPredictor",
]