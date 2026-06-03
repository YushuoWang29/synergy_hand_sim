from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import os, math

W, H = 2400, 1450
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

# =========================================================
# Fonts  -- 优先使用中文字体
# =========================================================
def find_font(candidates):
    for c in candidates:
        try:
            p = fm.findfont(c, fallback_to_default=False)
            if os.path.exists(p):
                return p
        except:
            pass
    return fm.findfont("DejaVu Sans")

font_path = find_font([
    "Microsoft YaHei",          # 微软雅黑
    "SimHei",                   # 黑体
    "Noto Sans SC",
    "SimSun",                   # 宋体
    "DejaVu Sans"
])

title_font = ImageFont.truetype(font_path, 46)
layer_font = ImageFont.truetype(font_path, 28)
main_font = ImageFont.truetype(font_path, 34)
sub_font = ImageFont.truetype(font_path, 24)

# =========================================================
# Colors
# =========================================================
BLACK = (20,20,20)
GRAY = (90,90,90)
LIGHT = (245,247,250)
BLUE = (55,95,180)

# =========================================================
# Helper Functions
# =========================================================
def center_text(text, box, font, fill=BLACK, spacing=8):
    x1,y1,x2,y2 = box
    lines = text.split("\n")

    heights = []
    total_h = 0

    for l in lines:
        b = draw.textbbox((0,0), l, font=font)
        h = b[3]-b[1]
        heights.append(h)
        total_h += h

    total_h += spacing*(len(lines)-1)

    y = (y1+y2-total_h)/2

    for l,h in zip(lines, heights):
        b = draw.textbbox((0,0), l, font=font)
        w = b[2]-b[0]

        draw.text(
            ((x1+x2-w)/2, y),
            l,
            font=font,
            fill=fill
        )

        y += h + spacing


def box(x1,y1,x2,y2,title,subtitle=None):

    draw.rounded_rectangle(
        [x1,y1,x2,y2],
        radius=24,
        fill=LIGHT,
        outline=BLACK,
        width=3
    )

    center_text(
        title,
        (x1+20,y1+15,x2-20,y1+80),
        main_font
    )

    if subtitle:
        center_text(
            subtitle,
            (x1+20,y1+80,x2-20,y2-15),
            sub_font,
            fill=GRAY
        )


def arrow(x1,y1,x2,y2,label=None, label_y_shift=0):

    draw.line(
        [x1,y1,x2,y2],
        fill=BLUE,
        width=5
    )

    ang = math.atan2(y2-y1, x2-x1)

    size = 18

    p1 = (x2,y2)

    p2 = (
        x2 - size*math.cos(ang-math.pi/6),
        y2 - size*math.sin(ang-math.pi/6)
    )

    p3 = (
        x2 - size*math.cos(ang+math.pi/6),
        y2 - size*math.sin(ang+math.pi/6)
    )

    draw.polygon([p1,p2,p3], fill=BLUE)

    if label:

        mx,my = (x1+x2)/2, (y1+y2)/2
        my += label_y_shift

        b = draw.textbbox((0,0), label, font=sub_font)

        w = b[2]-b[0]
        h = b[3]-b[1]

        draw.rounded_rectangle(
            [mx-w/2-10,my-h/2-5,mx+w/2+10,my+h/2+5],
            radius=8,
            fill="white"
        )

        draw.text(
            (mx-w/2,my-h/2),
            label,
            font=sub_font,
            fill=BLUE
        )

# =========================================================
# Title
# =========================================================
title = "SDAS 折纸灵巧手仿真框架"

b = draw.textbbox((0,0), title, font=title_font)

draw.text(
    ((W-(b[2]-b[0]))/2, 35),
    title,
    font=title_font,
    fill=BLACK
)

# =========================================================
# Layer Labels
# =========================================================
layers = [
    ("设计层", 130),
    ("数据模型层", 350),
    ("传动层", 570),
    ("协同求解层", 790),
    ("仿真与可视化", 1010),
]

for text, y in layers:

    draw.rounded_rectangle(
        [40,y,320,y+120],
        radius=18,
        outline=BLACK,
        width=3,
        fill="white"
    )

    center_text(
        text,
        (50,y+20,310,y+100),
        layer_font
    )

# =========================================================
# Main Blocks
# =========================================================

box(
    390,120,860,250,
    "Origami Hand CAD",
    "PyQt5-based graphical design editor"
)

box(
    1010,120,1420,250,
    ".ohd 文件",
    "JSON 几何与拓扑数据"
)

# right-shifted URDF Exporter
box(
    1750,120,2280,250,
    "URDF 导出器",
    "MuJoCo / 机器人模型生成"
)

# -- right-shifted left column (centered at x=1215, width=600) --
left_x1, left_x2 = 915, 1515

box(
    left_x1,340,left_x2,490,
    "OrigamiHandDesign",
    "fold_lines / faces /\n"
    "tendons / pulleys / actuators"
)

box(
    left_x1,560,left_x2,710,
    "TransmissionBuilder",
    "compute_R_one_sided()\n"
    "计算传动向量 R_A 和 R_B"
)

box(
    left_x1,780,left_x2,930,
    "SDASModel",
    "计算协同向量 S_A 和 S_B"
)

box(
    left_x1,1000,left_x2,1150,
    "MuJoCoSimulator",
    "实时交互物理仿真"
)

# -- right-shifted GLFW Viewer (center x=2015, width=600) --
box(
    1715,1000,2315,1150,
    "GLFW 查看器",
    "3D 渲染 / 滑块 / 关节监测"
)

# =========================================================
# Arrows
# =========================================================

# Top row
arrow(860,185,1010,185,"保存")
arrow(1420,185,1750,185,"导出")

# .ohd → OrigamiHandDesign (vertical down)
arrow(1215,250,1215,340,"加载")

# Left column vertical arrows
arrow(1215,490,1215,560)
arrow(1215,710,1215,780,"R_A  R_B", label_y_shift=-12)
arrow(1215,930,1215,1000,"S_A  S_B", label_y_shift=-12)

# URDF Exporter → GLFW Viewer (vertical down from bottom center)
arrow(2015,250,2015,1000,"URDF")

# MuJoCoSimulator → GLFW Viewer (horizontal callback)
arrow(left_x2,1075,1715,1075,"回调")

# =========================================================
# Footer
# =========================================================

footer = (
    "synergy_hand_sim · "
    "状态依赖自适应协同框架"
)

b = draw.textbbox((0,0), footer, font=sub_font)

draw.text(
    ((W-(b[2]-b[0]))/2, 1400),
    footer,
    font=sub_font,
    fill=GRAY
)

# =========================================================
# Save
# =========================================================

img.save("SDAS_software_architecture.png")

print("已保存图片：SDAS_software_architecture.png")
