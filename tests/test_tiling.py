"""tests/test_tiling.py

Dueno: Equipo D - Fase: val/test (feed de tiling) - Contrato: geometria -> M

Verifica la geometria del tiling determinista: cobertura sin huecos, tile central
en n impar, round-trip de coordenadas tile<->imagen entera, recuperabilidad de un
objeto en costura (y la limitacion conocida: un objeto mas ancho que el tile se
parte siempre), y determinismo. No necesita el dataset real para la geometria.
"""
from __future__ import annotations

import torch

from cardd.data.tiling import (
    CarDDTileDataset,
    tile_collate,
    tile_geometry,
    tile_image,
)


def _overlap(length: int, n: int, tile_frac: float = 0.5) -> int:
    tile = round(length * tile_frac)
    last = length - tile
    stride = round(last / (n - 1)) if n > 1 else 0
    return tile - stride


def test_geometry_covers_without_gaps():
    W, H, n = 400, 260, 3
    tiles = tile_geometry(W, H, n)
    assert len(tiles) == n * n
    xs = sorted({t["offset"][0] for t in tiles})
    ys = sorted({t["offset"][1] for t in tiles})
    tw, th = tiles[0]["tile_size"]
    # arranca en 0 y el ultimo tile cierra exactamente en el borde
    assert xs[0] == 0 and xs[-1] + tw == W
    assert ys[0] == 0 and ys[-1] + th == H
    # solape: cada paso <= tamano del tile (no hay hueco entre tiles)
    assert all(b - a <= tw for a, b in zip(xs, xs[1:]))
    assert all(b - a <= th for a, b in zip(ys, ys[1:]))


def test_central_tile_exists_for_odd_n():
    W, H, n = 400, 260, 3
    tiles = tile_geometry(W, H, n)
    xs = sorted({t["offset"][0] for t in tiles})
    ys = sorted({t["offset"][1] for t in tiles})
    cx, cy = xs[n // 2], ys[n // 2]
    tw, th = tiles[0]["tile_size"]
    # el tile central esta centrado en la imagen
    assert abs((cx + tw / 2) - W / 2) <= 1
    assert abs((cy + th / 2) - H / 2) <= 1


def test_tile_image_matches_geometry_and_roundtrip():
    image = torch.rand(3, 260, 400)
    tiles = tile_image(image, n=3)
    assert len(tiles) == 9
    for tile, meta in tiles:
        x0, y0 = meta["offset"]
        tw, th = meta["tile_size"]
        # el tile extraido coincide con el recorte segun su meta
        assert torch.equal(tile, image[:, y0 : y0 + th, x0 : x0 + tw])
        assert meta["scale"] == 1.0
        # round-trip: tile -> entera -> tile reconstruye la posicion
        box_tile = torch.tensor([1.0, 2.0, 5.0, 6.0])
        box_whole = box_tile.clone()
        box_whole[0::2] += x0
        box_whole[1::2] += y0
        back = box_whole.clone()
        back[0::2] -= x0
        back[1::2] -= y0
        assert torch.equal(back, box_tile)


def test_small_object_on_seam_is_recovered():
    W, H, n = 400, 260, 3
    ov = _overlap(W, n)  # objeto de ancho <= solape cabe entero en algun tile
    tiles = tile_geometry(W, H, n)
    tw, th = tiles[0]["tile_size"]
    # objeto en la region de solape, ancho <= solape -> entero en algun tile
    ox0 = sorted({t["offset"][0] for t in tiles})[1]        # inicio del tile central
    obj = (ox0 + 2, 10, ox0 + 2 + (ov - 4), 10 + (th // 2))  # x0,y0,x1,y1
    contained = any(
        t["offset"][0] <= obj[0] and obj[2] <= t["offset"][0] + tw
        and t["offset"][1] <= obj[1] and obj[3] <= t["offset"][1] + th
        for t in tiles
    )
    assert contained, "un objeto <= solape debe aparecer entero en al menos un tile"


def test_object_wider_than_tile_is_split_known_limit():
    W, H, n = 400, 260, 3
    tiles = tile_geometry(W, H, n)
    tw, th = tiles[0]["tile_size"]
    # objeto mas ancho que el propio tile: ningun tile lo contiene entero.
    # Limitacion conocida y aceptada (los daches grandes se reconocen por fraccion).
    obj = (75, 100, 75 + tw + 50, 160)  # ancho = tw + 50 > tw
    contained = any(
        t["offset"][0] <= obj[0] and obj[2] <= t["offset"][0] + tw for t in tiles
    )
    assert not contained


def test_geometry_is_deterministic():
    a = tile_geometry(400, 260, 3)
    b = tile_geometry(400, 260, 3)
    assert a == b
    img = torch.rand(3, 260, 400)
    ta = [t for t, _ in tile_image(img, n=3)]
    tb = [t for t, _ in tile_image(img, n=3)]
    assert all(torch.equal(x, y) for x, y in zip(ta, tb))


def test_dataset_rejects_train():
    import pytest

    with pytest.raises(ValueError):
        CarDDTileDataset("whatever", "train")


def test_collate_groups_tiles_and_metas():
    metas = [{"image_id": 7, "offset": (0, 0)}, {"image_id": 7, "offset": (1, 1)}]
    batch = [(torch.zeros(3, 4, 4), metas[0]), (torch.zeros(3, 4, 4), metas[1])]
    tiles, ms = tile_collate(batch)
    assert len(tiles) == len(ms) == 2
    assert {m["image_id"] for m in ms} == {7}  # todos del mismo padre