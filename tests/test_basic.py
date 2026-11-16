# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shadow.core import DepthMap, Vec3, light_view, load_scene  # noqa: E402


def test_depth_map_init():
    dm = DepthMap(16)
    assert dm.size == 16
    assert len(dm.data) == 256
    assert dm.at(3, 4) > 1e8


def test_light_view_orthonormal():
    d = Vec3(0.3, -1, 0.2)
    r, u, f = light_view(d)
    assert abs(r.dot(u)) < 1e-9
    assert abs(r.dot(f)) < 1e-9
    assert abs(u.dot(f)) < 1e-9
    assert abs(abs(r.cross(u).dot(f)) - 1.0) < 1e-9


def test_load_scene():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "scene.txt")
    if not os.path.exists(path):
        return
    scene = load_scene(path)
    assert len(scene["triangles"]) == 6
    assert scene["light"].y < 0
