# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shadow.core import DepthMap, Vec3, light_view, load_scene  # noqa: E402
from shadow.core import render_depth, shadow_factor, save_depth_png  # noqa: E402


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


def _flat_tri(y=0.0):
    return (Vec3(-0.5, y, -0.5), Vec3(0.5, y, -0.5), Vec3(0.0, y, 0.5))


def test_triangle_interior_filled():
    # 三角形内部必须有深度：骨架只写顶点时填充像素 <= 3
    dm = render_depth([_flat_tri(0.0)], Vec3(0, -1, 0), 64)
    filled = [v for v in dm.data if v < 1e8]
    assert len(filled) > 1000
    for v in filled:
        assert abs(v) < 1e-9  # 水平三角形深度处处为 0


def test_interpolated_depth_matches_plane():
    # 斜三角形：质心处深度应等于顶点深度的重心插值
    tri = (Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 1))
    light = Vec3(0, -1, 0)
    dm = render_depth([tri], light, 64)
    r, u, f = light_view(light)
    sx, sy = dm.project(Vec3(1.0 / 3, 1.0 / 3, 1.0 / 3), r, u)
    got = dm.at(int(round(sx)), int(round(sy)))
    assert got < 1e8
    tol = 2.0 * dm.half / dm.size  # 一个纹素内的插值误差
    assert abs(got - (-1.0 / 3)) < tol


def test_nearest_depth_wins():
    near = _flat_tri(1.0)  # 更靠近光源（上方），深度 -1
    far = _flat_tri(0.0)
    dm = render_depth([far, near], Vec3(0, -1, 0), 64)
    assert abs(dm.at(32, 32) - (-1.0)) < 1e-9
    dm2 = render_depth([near, far], Vec3(0, -1, 0), 64)  # 与三角形顺序无关
    assert abs(dm2.at(32, 32) - (-1.0)) < 1e-9


def test_out_of_bounds_sampling_clamps():
    dm = DepthMap(8)
    dm.set(7, 3, -100.0)  # 最后一列放一个近遮挡物
    light = Vec3(0, -1, 0)
    # 投影到 sx=38.5，远超宽度 8；钳制后应读到 (7, 3) 判为阴影。
    # 若不钳制，线性下标会读到别的行（1e9）而误判为受光。
    pos = Vec3(-1.0 / 7.0, 5.0, 10.0)
    assert shadow_factor(dm, pos, light, radius=0) == 0.0
    # 与直接采样边界点结果一致
    edge = Vec3(-1.0 / 7.0, 5.0, 1.0)  # sx 正好为 7
    assert shadow_factor(dm, edge, light, radius=0) == 0.0
    # 极远坐标不抛异常
    assert shadow_factor(dm, Vec3(0, 5.0, 10000.0), light, radius=0) in (0.0, 1.0)


def test_light_scale_and_reversal_invariant():
    tris = [_flat_tri(0.0), _flat_tri(1.0)]
    d1 = render_depth(tris, Vec3(0.3, -1, 0.2), 32)
    d2 = render_depth(tris, Vec3(0.75, -2.5, 0.5), 32)  # 同向不同模
    d3 = render_depth(tris, Vec3(-0.3, 1, -0.2), 32)    # 反向
    assert d1.data == d3.data  # 反向：深度符号不翻转，逐值相同
    assert max(abs(a - b) for a, b in zip(d1.data, d2.data)) < 1e-9
    pos = Vec3(0.1, 0.5, 0.1)
    assert (shadow_factor(d1, pos, Vec3(0.3, -1, 0.2))
            == shadow_factor(d3, pos, Vec3(-0.3, 1, -0.2)))


def test_light_near_parallel_to_up():
    # 光轴与默认 up 接近平行时基不能退化
    r, u, f = light_view(Vec3(0, -1, 1e-9))
    assert abs(r.dot(r) - 1.0) < 1e-6
    assert abs(u.dot(u) - 1.0) < 1e-6
    assert abs(r.dot(u)) < 1e-9
    assert abs(r.dot(f)) < 1e-9


def test_angle_perturbation_keeps_verdict():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene = load_scene(os.path.join(root, "data", "scene.txt"))
    tris = scene["triangles"]
    inside = Vec3(0, 0, -0.8)      # 挡板 1 的阴影里
    outside = Vec3(-1.5, 0, 1.5)   # 空旷处
    for light in (Vec3(0.3, -1, 0.2), Vec3(0.32, -1, 0.19), Vec3(0.28, -1, 0.21)):
        dm = render_depth(tris, light, 256)
        assert shadow_factor(dm, inside, light) < 0.5
        assert shadow_factor(dm, outside, light) > 0.5


def test_scene_floor_shadow():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene = load_scene(os.path.join(root, "data", "scene.txt"))
    light = scene["light"]
    dm = render_depth(scene["triangles"], light, 256)
    assert shadow_factor(dm, Vec3(0, 0, -0.8), light) < 0.5
    assert shadow_factor(dm, Vec3(-1.5, 0, 1.5), light) > 0.5


def test_png_deterministic(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene = load_scene(os.path.join(root, "data", "scene.txt"))
    dm1 = render_depth(scene["triangles"], scene["light"], 64)
    dm2 = render_depth(scene["triangles"], scene["light"], 64)
    assert dm1.data == dm2.data
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    save_depth_png(dm1, str(p1))
    save_depth_png(dm2, str(p2))
    assert p1.read_bytes() == p2.read_bytes()
