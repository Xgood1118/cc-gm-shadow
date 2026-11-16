# -*- coding: utf-8 -*-
"""阴影贴图骨架。

深度图按最近距离写入，但采样时不比较深度，也不做边界钳制。
"""

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


def light_view(light_dir, up=None):
    """光源方向的视图基。返回 (right, up, forward)。"""
    f = Vec3(-light_dir.x, -light_dir.y, -light_dir.z).normalized()
    if up is None:
        up = Vec3(0, 1, 0)
    r = f.cross(up).normalized()
    u = r.cross(f).normalized()
    return r, u, f


class DepthMap(object):
    def __init__(self, size):
        self.size = int(size)
        self.data = [1e9] * (self.size * self.size)

    def at(self, x, y):
        return self.data[y * self.size + x]

    def set(self, x, y, v):
        self.data[y * self.size + x] = v


def render_depth(triangles, light_dir, size):
    """把三角形投影到光源空间，写最近深度。当前不做任何偏移。"""
    r, u, f = light_view(light_dir)
    dm = DepthMap(size)
    for tri in triangles:
        for p in tri:
            depth = p.dot(f)
            sx = int(round((p.dot(r) * 0.5 + 0.5) * (size - 1)))
            sy = int(round((p.dot(u) * 0.5 + 0.5) * (size - 1)))
            if 0 <= sx < size and 0 <= sy < size:
                if depth < dm.at(sx, sy):
                    dm.set(sx, sy, depth)
    return dm


def shadow_factor(dm, pos, light_dir):
    """查询某点是否在阴影。当前不比较深度，也不做边界钳制。"""
    r, u, f = light_view(light_dir)
    sx = int(round((pos.dot(r) * 0.5 + 0.5) * (dm.size - 1)))
    sy = int(round((pos.dot(u) * 0.5 + 0.5) * (dm.size - 1)))
    depth = pos.dot(f)
    stored = dm.at(sx, sy)
    return 0.0 if stored < depth else 1.0


def save_depth_png(dm, path):
    """把深度可视化。"""
    raw = bytearray()
    lo = min(dm.data)
    hi = max(dm.data)
    span = (hi - lo) or 1.0
    for y in range(dm.size):
        raw.append(0)
        for x in range(dm.size):
            v = int(255 * (1.0 - (dm.at(x, y) - lo) / span))
            raw.extend((v, v, v, 255))
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


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="shadow")
    ap.add_argument("scene")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="depth.png")
    args = ap.parse_args(argv)
    scene = load_scene(args.scene)
    dm = render_depth(scene["triangles"], scene["light"], args.size)
    save_depth_png(dm, args.out)
    print("深度图 %d，三角形 %d" % (args.size, len(scene["triangles"])))
    return 0
