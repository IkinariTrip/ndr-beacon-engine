"""Incidence Cockpit アイコン生成（宇宙要塞シルエット）"""
import os, random, sys
from PIL import Image, ImageDraw

S, N = 4, 512


def P(pts):
    return [(x * S, y * S) for x, y in pts]


def build():
    img = Image.new("RGBA", (N * S, N * S), (0, 0, 0, 0))
    bg = Image.new("RGBA", (N * S, N * S))
    d = ImageDraw.Draw(bg)
    for y in range(N * S):
        t = y / (N * S)
        d.line([(0, y), (N * S, y)], fill=(int(14 + 36 * t), int(24 + 44 * t), int(52 + 66 * t), 255))
    mask = Image.new("L", (N * S, N * S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, N * S - 1, N * S - 1], radius=96 * S, fill=255)
    img.paste(bg, (0, 0), mask)
    d = ImageDraw.Draw(img)
    random.seed(7)
    for _ in range(26):
        x, y, r = random.randint(24, N - 24), random.randint(24, N - 24), random.choice([1, 1, 2])
        d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=(220, 230, 255, 200))
    x, y = 430, 96
    d.polygon(P([(x, y - 14), (x + 3, y - 3), (x + 14, y), (x + 3, y + 3), (x, y + 14),
                 (x - 3, y + 3), (x - 14, y), (x - 3, y - 3)]), fill=(235, 240, 255, 255))

    knob = [(242, 160), (246, 138), (238, 130), (240, 118), (272, 118), (274, 130), (266, 138), (270, 160)]
    disc = [(30, 224), (70, 212), (110, 196), (150, 180), (200, 164), (256, 156), (312, 164), (362, 180),
            (402, 196), (442, 212), (482, 224), (452, 240), (410, 262), (370, 284), (330, 302), (292, 316),
            (256, 320), (220, 316), (182, 302), (142, 284), (102, 262), (60, 240)]
    tail = [(200, 334), (256, 324), (318, 334), (302, 356), (290, 386), (288, 398), (336, 408), (292, 418),
            (296, 434), (348, 468), (302, 458), (286, 474), (278, 502), (264, 452), (256, 430), (244, 414),
            (188, 406), (240, 394), (230, 372), (218, 354)]
    base, rim = (18, 28, 46, 255), (86, 158, 206, 255)

    def shape(pts, fill, edge):
        d.polygon(P(pts), fill=fill)
        d.line(P(pts + [pts[0]]), fill=edge, width=3 * S, joint="curve")

    shape(knob, (40, 64, 92, 255), (120, 190, 230, 255))
    shape(disc, base, rim)
    shape(tail, base, rim)
    for xx in range(90, 430, 24):
        t = (xx - 256) / 200
        d.line(P([(xx, 228 + abs(t) * 8), (256 + t * 34, 312)]), fill=(34, 56, 84, 255), width=2 * S)
    for xx in range(70, 450, 22):
        yy = 226 + abs(xx - 256) * 0.04
        d.rectangle([xx * S, yy * S, (xx + 4) * S, (yy + 2.5) * S], fill=(255, 222, 140, 255))
    return img.resize((N, N), Image.LANCZOS)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "assets/incidence_cockpit_icon.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    build().save(out)
    print("アイコンを作成しました:", out)
