"""Tests de tile_fusion.fuse_tile_predictions_graph. Sin modelo, sin GPU:
mascaras sinteticas. Geometria: imagen 100x80, tile 50x40."""
import numpy as np
import torch

from cardd.engine.tile_fusion import fuse_tile_predictions_graph

W, H = 100, 80
TW, TH = 50, 40


def meta(x0, y0):
    return {"image_id": 1, "offset": (x0, y0), "tile_size": (TW, TH), "orig_size": (W, H), "scale": 1.0}


def tile_out(rects, labels, scores):
    """rects: lista de (x0,y0,x1,y1) en coords LOCALES del tile -> mascaras 1.0."""
    n = len(rects)
    masks = torch.zeros(n, 1, TH, TW, dtype=torch.float32)
    for k, (x0, y0, x1, y1) in enumerate(rects):
        masks[k, 0, y0:y1, x0:x1] = 1.0
    return {
        "boxes": torch.zeros(n, 4),  # no se usa
        "labels": torch.tensor(labels, dtype=torch.int64),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "masks": masks,
    }


CFG = {"match_metric": "containment", "match_thr": 0.5, "mask_weight": "uniform", "mask_thr": 0.5, "score_reduce": "max"}


def test_single_detection_box_from_mask():
    # objeto entero en tile (0,0), local x10..30 y5..25 -> whole igual (offset 0)
    outs = [tile_out([(10, 5, 30, 25)], [2], [0.9])]
    metas = [meta(0, 0)]
    r = fuse_tile_predictions_graph(outs, metas, CFG)
    assert r["boxes"].shape[0] == 1
    assert list(r["boxes"][0].numpy()) == [10, 5, 30, 25]
    assert int(r["labels"][0]) == 2
    assert abs(float(r["scores"][0]) - 0.9) < 1e-6
    assert r["masks"].shape == (1, 1, H, W)


def test_seam_object_merges_into_one():
    # tiles solapados (0,0) y (25,0). Objeto whole x30..60 y10..30.
    # A ve x30..50 (local 30..50), B ve x30..60 (local 5..35).
    a = tile_out([(30, 10, 50, 30)], [3], [0.8])
    b = tile_out([(5, 10, 35, 30)], [3], [0.6])
    r = fuse_tile_predictions_graph([a, b], [meta(0, 0), meta(25, 0)], CFG)
    assert r["boxes"].shape[0] == 1, "el objeto de costura debe quedar como UNA instancia"
    # caja fusionada cubre whole x30..60 y10..30
    assert list(r["boxes"][0].numpy()) == [30, 10, 60, 30]
    assert abs(float(r["scores"][0]) - 0.8) < 1e-6, "score = maximo del grupo"


def test_distinct_same_class_not_merged():
    # tiles SIN solape (0,0) y (50,0), dos objetos misma clase lejos -> 2 instancias
    a = tile_out([(5, 10, 15, 20)], [1], [0.9])      # whole x5..15
    b = tile_out([(10, 10, 20, 20)], [1], [0.7])     # whole x60..70
    r = fuse_tile_predictions_graph([a, b], [meta(0, 0), meta(50, 0)], CFG)
    assert r["boxes"].shape[0] == 2, "objetos distintos misma clase no se fusionan"


def test_same_tile_never_merges():
    # dos detecciones misma clase en el MISMO tile -> nunca se fusionan
    a = tile_out([(2, 2, 12, 12), (30, 20, 45, 35)], [1, 1], [0.9, 0.8])
    r = fuse_tile_predictions_graph([a], [meta(0, 0)], CFG)
    assert r["boxes"].shape[0] == 2


def test_different_class_not_merged():
    # mismo lugar, clases distintas -> no se fusionan
    a = tile_out([(30, 10, 50, 30)], [3], [0.8])
    b = tile_out([(5, 10, 35, 30)], [4], [0.6])
    r = fuse_tile_predictions_graph([a, b], [meta(0, 0), meta(25, 0)], CFG)
    assert r["boxes"].shape[0] == 2


def test_containment_vs_iou_diverge():
    # fragmento chico (viruta de costura) vs mascara grande.
    # A: whole x30..50 y10..30 (area 400). B: whole x45..50 y10..14 (area 20, viruta).
    a = tile_out([(30, 10, 50, 30)], [2], [0.9])            # tile(0,0)
    b = tile_out([(20, 10, 25, 14)], [2], [0.5])            # tile(25,0): local x20..25 -> whole x45..50
    metas = [meta(0, 0), meta(25, 0)]
    # containment: inter/min = 20/20 = 1.0 -> une
    r_cont = fuse_tile_predictions_graph([a, b], metas, {**CFG, "match_metric": "containment", "match_thr": 0.5})
    # iou: inter/union = 20/400 = 0.05 -> con thr 0.5 NO une
    r_iou = fuse_tile_predictions_graph([a, b], metas, {**CFG, "match_metric": "iou", "match_thr": 0.5})
    assert r_cont["boxes"].shape[0] == 1, "containment une la viruta con la mascara grande"
    assert r_iou["boxes"].shape[0] == 2, "iou con thr alto deja la viruta como instancia aparte"


def test_score_reduce_variants():
    a = tile_out([(30, 10, 50, 30)], [3], [0.8])
    b = tile_out([(5, 10, 35, 30)], [3], [0.6])
    metas = [meta(0, 0), meta(25, 0)]
    r_max = fuse_tile_predictions_graph([a, b], metas, {**CFG, "score_reduce": "max"})
    r_wavg = fuse_tile_predictions_graph([a, b], metas, {**CFG, "score_reduce": "wavg_area"})
    assert abs(float(r_max["scores"][0]) - 0.8) < 1e-6
    # wavg: areas 400 (0.8) y 600 (0.6) -> (400*.8+600*.6)/1000 = 0.68
    assert abs(float(r_wavg["scores"][0]) - 0.68) < 1e-6


def test_empty_input():
    a = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64),
         "scores": torch.zeros(0), "masks": torch.zeros(0, 1, TH, TW)}
    r = fuse_tile_predictions_graph([a], [meta(0, 0)], CFG)
    assert r["boxes"].shape == (0, 4)
    assert r["masks"].shape == (0, 1, H, W)


def test_deterministic():
    a = tile_out([(30, 10, 50, 30)], [3], [0.8])
    b = tile_out([(5, 10, 35, 30)], [3], [0.6])
    metas = [meta(0, 0), meta(25, 0)]
    r1 = fuse_tile_predictions_graph([a, b], metas, CFG)
    r2 = fuse_tile_predictions_graph([a, b], metas, CFG)
    assert torch.equal(r1["boxes"], r2["boxes"])
    assert torch.equal(r1["masks"], r2["masks"])


def test_feathered_runs_and_centers_higher():
    # feathered debe correr y dar mascara valida; el peso central > peso de borde
    a = tile_out([(0, 0, 50, 40)], [1], [0.9])  # tile lleno
    r = fuse_tile_predictions_graph([a], [meta(0, 0)], {**CFG, "mask_weight": "feathered"})
    assert r["boxes"].shape[0] == 1
    assert float(r["masks"].max()) <= 1.0 + 1e-6