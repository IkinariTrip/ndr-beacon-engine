import os, random, sys
from PIL import Image, ImageDraw

S, N = 4, 512

def P(pts):
    return [(x * S, y * S) for x, y in pts]

def build():
    img = Image.new("RGBA", (N * S, N * S), (0, 0, 0, 0))
    bg = Image.new("RGBA", (N * S, N * S))
    d = ImageDraw.Draw(bg)
    
    # 背景グラデーション（白〜非常に薄い水色）
    for y in range(N * S):
        t = y / (N * S)
        d.line([(0, y), (N * S, y)], fill=(int(255 - 10*t), int(255 - 5*t), 255, 255))
        
    mask = Image.new("L", (N * S, N * S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, N * S - 1, N * S - 1], radius=96 * S, fill=255)
    img.paste(bg, (0, 0), mask)
    
    d = ImageDraw.Draw(img)
    random.seed(7)
    
    # 星（濃い青）
    for _ in range(26):
        x, y, r = random.randint(24, N - 24), random.randint(24, N - 24), random.choice([1, 1, 2])
        d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=(10, 30, 80, 150))
        
    # 戦艦の先端（濃紺）
    x, y = 430, 96
    d.polygon(P([(x, y - 14), (x + 3, y - 3), (x + 14, y), (x + 3, y + 3), (x, y + 14),
                 (x - 3, y + 3), (x - 14, y), (x - 3, y - 3)]), fill=(20, 30, 80, 255))
                 
    knob = [(242, 160), (246, 138), (238, 130), (240, 118), (272, 118), (274, 130), (266, 138), (270, 160)]
    disc = [(30, 224), (70, 212), (110, 196), (150, 180), (200, 164), (256, 156), (312, 164), (362, 180),
            (402, 196), (442, 212), (482, 224), (452, 240), (410, 262), (370, 284), (330, 302), (292, 316),
            (256, 320), (220, 316), (182, 302), (142, 284), (102, 262), (60, 240)]
    tail = [(200, 334), (256, 324), (318, 334), (302, 356), (290, 386), (288, 398), (336, 408), (292, 418),
            (296, 434), (348, 468), (302, 458), (286, 474), (278, 502), (264, 452), (256, 430), (244, 414),
            (188, 406), (240, 394), (230, 372), (218, 354)]
            
    # 戦艦のベース（青系）と縁取り（濃紺）
    base, rim = (60, 90, 160, 255), (10, 20, 60, 255)
    
    def shape(pts, fill, edge):
        d.polygon(P(pts), fill=fill)
        d.line(P(pts + [pts[0]]), fill=edge, width=3 * S, joint="curve")
        
    shape(knob, (100, 130, 200, 255), (20, 40, 80, 255))
    shape(disc, base, rim)
    shape(tail, base, rim)
    
    # 模様の線（白っぽく）
    for xx in range(90, 430, 24):
        t = (xx - 256) / 200
        d.line(P([(xx, 228 + abs(t) * 8), (256 + t * 34, 312)]), fill=(220, 230, 255, 255), width=2 * S)
        
    # エンジンの光（濃いオレンジ）
    for xx in range(70, 450, 22):
        yy = 226 + abs(xx - 256) * 0.04
        d.rectangle([xx * S, yy * S, (xx + 4) * S, (yy + 2.5) * S], fill=(255, 140, 0, 255))
        
    return img.resize((N, N), Image.LANCZOS)

out = "assets/incidence_cockpit_icon.png"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
build().save(out)
print("白背景・青戦艦の新しいアイコンを作成しました:", out)
