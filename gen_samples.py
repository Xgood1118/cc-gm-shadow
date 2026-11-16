# -*- coding: utf-8 -*-
"""一层地板加两块高矮不同的挡板。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SCENE = """light 0.3 -1 0.2
v -2 0 -2
v 2 0 -2
v 2 0 2
v -2 0 2
v -1 0 -1
v 1 0 -1
v 1 2 -1
v -1 2 -1
v 0.5 0 0.5
v 1.5 0 0.5
v 1.5 1 0.5
v 0.5 1 0.5
tri 0 1 2
tri 0 2 3
tri 4 5 6
tri 4 6 7
tri 8 9 10
tri 8 10 11
"""


def main():
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    path = os.path.join(HERE, "data", "scene.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(SCENE)
    print("已生成", path)


if __name__ == "__main__":
    main()
