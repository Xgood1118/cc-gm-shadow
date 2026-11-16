# -*- coding: utf-8 -*-
"""阴影贴图：光源正交深度图 + 深度比较 + PCF 软阴影。

约定：
- ``light_dir`` 为光线传播方向（从光源指向场景），模长任意。
- 光轴做了符号规范化：``light_dir`` 整体取反时视图基不变，
  深度符号不翻转，阴影判定结果保持一致。
- 深度 = 点在光轴（forward）上的投影坐标，值越小离光源越近；
  深度图逐像素保留最小值（最近遮挡物）。
- 采样坐标一律钳制到深度图边界，越界按边界值处理。
"""

import math
import struct
import zlib


class Vec3(object):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def sub(self, o):
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o):
        return Vec3(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z,
                    self.x * o.y - self.y * o.x)

    def normalized(self):
        n = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
        if n == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / n, self.y / n, self.z / n)


def _canonical_axis(d):
    """把方向规范到固定半球：y、x、z 中第一个非零分量取负。

    d 与 -d 规范化后相同，因此光源方向写反时光轴不变。
    """
    if d.y > 0 or (d.y == 0 and (d.x > 0 or (d.x == 0 and d.z > 0))):
        return Vec3(-d.x, -d.y, -d.z)
    return d


def light_view(light_dir, up=None):
    """光源方向的视图基。返回 (right, up, forward)。

    forward 沿光线传播方向，并做符号规范化：light_dir 取反得到完全相同的基。
    light_dir 与 up 接近平行时自动换用备用参考轴，避免基退化。
    """
    f = Vec3(light_dir.x, light_dir.y, light_dir.z).normalized()
    f = _canonical_axis(f)
    if up is None:
        up = Vec3(0, 1, 0)
    r = f.cross(up)
    n = (r.x ** 2 + r.y ** 2 + r.z ** 2) ** 0.5
    if n < 1e-6:
        up = Vec3(1, 0, 0)
        r = f.cross(up)
    r = r.normalized()
    u = r.cross(f).normalized()
    return r, u, f


class DepthMap(object):
    def __init__(self, size):
        self.size = int(size)
        self.data = [1e9] * (self.size * self.size)
        # 正交投影参数：光轴垂直面内的中心 (center) 与半径 half（世界单位）。
        # 默认对应 NDC [-1, 1] 映射；render_depth 会按场景包围盒自适应。
        self.center = (0.0, 0.0)
        self.half = 1.0

    def at(self, x, y):
        return self.data[y * self.size + x]

    def set(self, x, y, v):
        self.data[y * self.size + x] = v

    def at_clamped(self, x, y):
        """钳制到边界后读取，越界采样按边界值处理。"""
        x = min(max(int(round(x)), 0), self.size - 1)
        y = min(max(int(round(y)), 0), self.size - 1)
        return self.data[y * self.size + x]

    def project(self, p, r, u):
        """世界点到连续像素坐标（像素中心位于整数坐标）。"""
        s = self.size - 1
        sx = (((p.dot(r) - self.center[0]) / self.half) * 0.5 + 0.5) * s
        sy = (((p.dot(u) - self.center[1]) / self.half) * 0.5 + 0.5) * s
        return sx, sy


def _fit_projection(triangles, r, u, margin=1.05):
    """按场景在光轴垂直面内的包围盒确定投影中心与半径。"""
    lo_r = hi_r = lo_u = hi_u = None
    for tri in triangles:
        for p in tri:
            pr, pu = p.dot(r), p.dot(u)
            if lo_r is None:
                lo_r = hi_r = pr
                lo_u = hi_u = pu
            else:
                lo_r = min(lo_r, pr)
                hi_r = max(hi_r, pr)
                lo_u = min(lo_u, pu)
                hi_u = max(hi_u, pu)
    if lo_r is None:
        return 0.0, 0.0, 1.0
    cr = (lo_r + hi_r) * 0.5
    cu = (lo_u + hi_u) * 0.5
    half = max(hi_r - lo_r, hi_u - lo_u) * 0.5 * margin
    if half <= 0:
        half = 1.0
    return cr, cu, half


def _rasterize(dm, triangles, r, u, f):
    """把三角形光栅化到深度图：内部按重心坐标插值深度，逐像素保留最近值。"""
    s = dm.size
    for tri in triangles:
        pts = []
        for p in tri:
            sx, sy = dm.project(p, r, u)
            pts.append((sx, sy, p.dot(f)))
        (x0, y0, d0), (x1, y1, d1), (x2, y2, d2) = pts
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if denom == 0:
            continue
        xa = max(0, int(math.floor(min(x0, x1, x2))))
        xb = min(s - 1, int(math.ceil(max(x0, x1, x2))))
        ya = max(0, int(math.floor(min(y0, y1, y2))))
        yb = min(s - 1, int(math.ceil(max(y0, y1, y2))))
        for y in range(ya, yb + 1):
            for x in range(xa, xb + 1):
                w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
                w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
                w2 = 1.0 - w0 - w1
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                depth = w0 * d0 + w1 * d1 + w2 * d2
                if depth < dm.at(x, y):
                    dm.set(x, y, depth)


def render_depth(triangles, light_dir, size):
    """把三角形光栅化到光源正交深度图，逐像素保留离光源最近的深度。

    投影范围按场景包围盒自适应（含 5% 边距），结果存在 dm.center/dm.half，
    shadow_factor 会用同一组参数投影，保证渲染与查询一致。
    """
    r, u, f = light_view(light_dir)
    dm = DepthMap(size)
    cr, cu, half = _fit_projection(triangles, r, u)
    dm.center = (cr, cu)
    dm.half = half
    _rasterize(dm, triangles, r, u, f)
    return dm


def shadow_factor(dm, pos, light_dir, radius=2, bias=None):
    """查询某点的受光比例：1.0 完全受光，0.0 完全在阴影中。

    判据：点的深度 <= 深度图存储深度 + bias 时视为受光，否则为遮挡。
    在投影点周围做 (2*radius+1)^2 次 PCF 采样取平均，radius 控制边缘
    过渡宽度；采样坐标钳制到深度图边界；bias 为自遮挡深度偏移，
    默认 4*half/size（约 2 个纹素的深度余量）。
    """
    r, u, f = light_view(light_dir)
    sx, sy = dm.project(pos, r, u)
    depth = pos.dot(f)
    if bias is None:
        bias = 4.0 * dm.half / dm.size
    rad = max(0, int(round(radius)))
    cx = int(round(sx))
    cy = int(round(sy))
    lit = 0
    total = 0
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            stored = dm.at_clamped(cx + dx, cy + dy)
            total += 1
            if depth <= stored + bias:
                lit += 1
    return lit / total


def save_depth_png(dm, path):
    """把深度可视化成 PNG（空像素为黑，有效深度按实际范围归一化）。"""
    finite = [v for v in dm.data if v < 1e8]
    lo = min(finite) if finite else 0.0
    hi = max(finite) if finite else 1.0
    span = (hi - lo) or 1.0
    raw = bytearray()
    for y in range(dm.size):
        raw.append(0)
        for x in range(dm.size):
            v = dm.at(x, y)
            g = 0 if v >= 1e8 else int(255 * (1.0 - (v - lo) / span))
            raw.extend((g, g, g, 255))

    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", dm.size, dm.size, 8, 6, 0, 0, 0)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(data)


def load_scene(path):
    verts = []
    tris = []
    light = Vec3(0, -1, 0)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "light":
                light = Vec3(float(parts[1]), float(parts[2]), float(parts[3]))
            elif parts[0] == "v":
                verts.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "tri":
                tris.append((verts[int(parts[1])], verts[int(parts[2])], verts[int(parts[3])]))
    return {"light": light, "triangles": tris}


def _floor_and_panels(triangles):
    """按最低 y 平面分出地板三角形，其余按共享顶点聚成挡板组。"""
    ymin = min(p.y for tri in triangles for p in tri)
    floor = [t for t in triangles if all(p.y == ymin for p in t)]
    rest = [t for t in triangles if t not in floor]
    groups = []
    for t in rest:
        ids = set(id(p) for p in t)
        merged = [t]
        remain = []
        for g in groups:
            gids = set(id(p) for tri2 in g for p in tri2)
            if gids & ids:
                merged.extend(g)
            else:
                remain.append(g)
        remain.append(merged)
        groups = remain
    return floor, groups, ymin


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="shadow")
    ap.add_argument("scene")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="depth.png")
    ap.add_argument("--softness", type=int, default=2,
                    help="PCF 采样半径（像素），控制阴影边缘过渡宽度")
    args = ap.parse_args(argv)
    scene = load_scene(args.scene)
    light = scene["light"]
    triangles = scene["triangles"]
    dm = render_depth(triangles, light, args.size)
    save_depth_png(dm, args.out)
    print("深度图 %d，三角形 %d" % (args.size, len(triangles)))

    floor, panels, ymin = _floor_and_panels(triangles)
    if not floor:
        return 0
    r, u, f = light_view(light)
    bias = 4.0 * dm.half / dm.size
    # 每块挡板单独渲染一张同投影的深度图，用于统计各自的投影面积
    panel_dms = []
    for group in panels:
        pdm = DepthMap(args.size)
        pdm.center = dm.center
        pdm.half = dm.half
        _rasterize(pdm, group, r, u, f)
        panel_dms.append(pdm)
    xs = [p.x for t in floor for p in t]
    zs = [p.z for t in floor for p in t]
    x0, x1 = min(xs), max(xs)
    z0, z1 = min(zs), max(zs)
    cell = ((x1 - x0) / args.size) * ((z1 - z0) / args.size)
    lit = 0
    shadow = 0
    hits = [0] * len(panel_dms)
    for iy in range(args.size):
        z = z0 + (z1 - z0) * (iy + 0.5) / args.size
        for ix in range(args.size):
            x = x0 + (x1 - x0) * (ix + 0.5) / args.size
            pos = Vec3(x, ymin, z)
            if shadow_factor(dm, pos, light, radius=args.softness) >= 0.5:
                lit += 1
            else:
                shadow += 1
            depth = pos.dot(f)
            sx, sy = dm.project(pos, r, u)
            px = int(round(sx))
            py = int(round(sy))
            for i, pdm in enumerate(panel_dms):
                if depth > pdm.at_clamped(px, py) + bias:
                    hits[i] += 1
    print("受光像素 %d，阴影像素 %d（共 %d）" % (lit, shadow, lit + shadow))
    for i, n in enumerate(hits):
        print("挡板 %d 投影：%d 像素，%.4f 平方单位" % (i + 1, n, n * cell))
    return 0
