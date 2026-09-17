#!/usr/bin/env python3
"""Generate dense and single-path dual-task progress diagrams.

Install: python3 -m pip install reportlab
PNG rendering additionally requires Poppler (pdftoppm).
Run:     python3 generate_figure.py

All coordinates, stroke widths, and font sizes below use the SVG viewBox.
The dense diagram contains 13 symmetric, monotone trajectories that nearly
cover the full square. The companion diagram keeps only one clearly visible
trajectory through the lower-right region. Every stroke uses a native gradient derived
from one shared, continuous 2-D OKLab color field. The PDF uses native axial
shadings clipped to smooth vector stroke outlines; it contains no raster
images. PNG is rendered from that same PDF.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from html import escape
from math import hypot
from pathlib import Path
import subprocess

import reportlab
from reportlab.lib.colors import Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


CANVAS = 560.0
LEFT, TOP, SIDE = 90.0, 58.0, 404.0
CURVE_WIDTH, AXIS_WIDTH = 2.8, 1.65
TRAJECTORY_COUNT = 13
POWER = 4
SINGLE_BEND = 1.0
DOT_RADIUS = 2.5
ARROW_LENGTH, ARROW_HALF_WIDTH = 7.5, 3.5
DRAWIO_STROKE_SCALE = 3.0
DRAWIO_FONT_SCALE = 1.5
DRAWIO_VIEWBOX = (24.0, 43.0, 537.0, 461.0)
GRADIENT_STEPS = 64
OUTLINE_STEPS = 48
SVG_CURVE_STEPS = 32
TEXT_COLOR, COORD_COLOR = "#626B7B", "#9299A6"

# Four anchors for a single spatial color field. No field background is drawn.
PALETTE = {
    "origin": "#8EA9E8",
    "task1": "#13B8A6",
    "task2": "#F28A5B",
    "joint": "#9A7BD8",
}

Vec = tuple[float, float]
RGB = tuple[float, float, float]


def rgb(hex_color: str) -> RGB:
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def to_oklab(color: RGB) -> RGB:
    r, g, b = (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055)**2.4
               for v in color)
    l = (0.4122214708*r + 0.5363325363*g + 0.0514459929*b)**(1/3)
    m = (0.2119034982*r + 0.6806995451*g + 0.1073969566*b)**(1/3)
    s = (0.0883024619*r + 0.2817188376*g + 0.6299787005*b)**(1/3)
    return (0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
            1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
            0.0259040371*l + 0.7827717662*m - 0.8086757660*s)


def from_oklab(color: RGB) -> RGB:
    lightness, a, b = color
    l = (lightness + 0.3963377774*a + 0.2158037573*b)**3
    m = (lightness - 0.1055613458*a - 0.0638541728*b)**3
    s = (lightness - 0.0894841775*a - 1.2914855480*b)**3
    linear = (4.0767416621*l - 3.3077115913*m + 0.2309699292*s,
              -1.2684380046*l + 2.6097574011*m - 0.3413193965*s,
              -0.0041960863*l - 0.7034186147*m + 1.7076147010*s)
    return tuple(min(1.0, max(0.0, 12.92*v if v <= 0.0031308
                            else 1.055*v**(1/2.4) - 0.055)) for v in linear)


LAB_CORNERS = tuple(to_oklab(rgb(PALETTE[name]))
                    for name in ("origin", "task1", "task2", "joint"))


def field(x: float, y: float) -> RGB:
    """Bilinear interpolation in OKLab: color depends on both task progresses."""
    weights = ((1-x)*(1-y), x*(1-y), (1-x)*y, x*y)
    return from_oklab(tuple(sum(w*c[k] for w, c in zip(weights, LAB_CORNERS))
                           for k in range(3)))


def svg_color(color: RGB) -> str:
    return "rgb(" + ",".join(f"{v * 100:.5f}%" for v in color) + ")"


def svg_point(point: Vec) -> Vec:
    x, y = point
    return LEFT + SIDE*x, TOP + SIDE*(1-y)


def pdf_point(point: Vec) -> Vec:
    x, y = svg_point(point)
    return x, CANVAS-y


def point_string(point: Vec) -> str:
    return f"{point[0]:.6f},{point[1]:.6f}"


@dataclass(frozen=True)
class Stroke:
    name: str
    width: float
    kind: str
    bend: float = 0.0

    @property
    def p0(self) -> Vec:
        return (0.0, 0.0)

    @property
    def p2(self) -> Vec:
        if self.kind == "task1-axis":
            return (1.0, 0.0)
        if self.kind == "task2-axis":
            return (0.0, 1.0)
        return (1.0, 1.0)

    def point(self, t: float) -> Vec:
        if self.kind != "trajectory":
            return tuple(t*v for v in self.p2)
        amount = abs(self.bend)
        slow = t**POWER
        fast = 1-(1-t)**POWER
        if self.bend >= 0:
            return ((1-amount)*t + amount*fast,
                    (1-amount)*t + amount*slow)
        return ((1-amount)*t + amount*slow,
                (1-amount)*t + amount*fast)

    def velocity(self, t: float) -> Vec:
        if self.kind != "trajectory":
            return self.p2
        amount = abs(self.bend)
        slow = POWER*t**(POWER-1)
        fast = POWER*(1-t)**(POWER-1)
        if self.bend >= 0:
            return ((1-amount) + amount*fast,
                    (1-amount) + amount*slow)
        return ((1-amount) + amount*slow,
                (1-amount) + amount*fast)

    def acceleration(self, t: float) -> Vec:
        if self.kind != "trajectory":
            return (0.0, 0.0)
        amount = abs(self.bend)
        slow = POWER*(POWER-1)*t**(POWER-2)
        fast = -POWER*(POWER-1)*(1-t)**(POWER-2)
        if self.bend >= 0:
            return (amount*fast, amount*slow)
        return (amount*slow, amount*fast)

    def gradient_offset(self, t: float) -> float:
        point = self.point(t)
        dx, dy = self.p2
        return (point[0]*dx + point[1]*dy)/(dx*dx + dy*dy)

    def outline(self, t: float, sign: int) -> tuple[Vec, Vec]:
        """Position and derivative of the offset curve, in PDF coordinates."""
        px, py = pdf_point(self.point(t))
        vx, vy = (SIDE*v for v in self.velocity(t))
        ax, ay = (SIDE*v for v in self.acceleration(t))
        speed = hypot(vx, vy)
        dot = vx*ax + vy*ay
        radius = sign*self.width/2
        normal = (-vy/speed, vx/speed)
        derivative = (-ay/speed + vy*dot/speed**3,
                      ax/speed - vx*dot/speed**3)
        return ((px+radius*normal[0], py+radius*normal[1]),
                (vx+radius*derivative[0], vy+radius*derivative[1]))


def geometry(single_lower_right: bool = False) -> list[Stroke]:
    axes = [Stroke("task1-axis", AXIS_WIDTH, "task1-axis"),
            Stroke("task2-axis", AXIS_WIDTH, "task2-axis")]
    bends = [SINGLE_BEND] if single_lower_right else [
        -1 + 2*index/(TRAJECTORY_COUNT-1)
        for index in range(TRAJECTORY_COUNT)
    ]
    curves = [Stroke(f"trajectory-{index}", CURVE_WIDTH, "trajectory", bend)
              for index, bend in enumerate(bends)]
    return axes + curves


def svg_path(stroke: Stroke) -> str:
    start = point_string(svg_point(stroke.point(0)))
    if stroke.kind != "trajectory":
        return f"M {start} L {point_string(svg_point(stroke.point(1)))}"
    commands = [f"M {start}"]
    for index in range(SVG_CURVE_STEPS):
        t0, t1 = index/SVG_CURVE_STEPS, (index+1)/SVG_CURVE_STEPS
        h = (t1-t0)/3
        p0, p1 = stroke.point(t0), stroke.point(t1)
        v0, v1 = stroke.velocity(t0), stroke.velocity(t1)
        c0 = tuple(p0[k]+h*v0[k] for k in range(2))
        c1 = tuple(p1[k]-h*v1[k] for k in range(2))
        commands.append("C " + " ".join(point_string(svg_point(point))
                                          for point in (c0, c1, p1)))
    return " ".join(commands)


def labels(drawio_compatible: bool = False) -> list[
        tuple[str, float, float, float, str, int, str]]:
    """Text, SVG baseline x/y, font size, anchor, SVG rotation, fill color."""
    task1_y = 494 if drawio_compatible else 526
    task2_x = 70 if drawio_compatible else 25
    return [
        ("Task 1 progress", LEFT+SIDE/2, task1_y, 22, "middle", 0, TEXT_COLOR),
        ("Task 2 progress", task2_x, TOP+SIDE/2, 22, "middle", -90, TEXT_COLOR),
        ("(0, 0)", LEFT-10, TOP+SIDE+30, 15, "end", 0, COORD_COLOR),
        ("(1, 0)", LEFT+SIDE, TOP+SIDE+30, 15, "middle", 0, COORD_COLOR),
        ("(0, 1)", LEFT-12, TOP+5, 15, "end", 0, COORD_COLOR),
        ("(1, 1)", LEFT+SIDE+13, TOP+5, 15, "start", 0, COORD_COLOR),
    ]


def arrowheads() -> list[tuple[list[Vec], RGB]]:
    x0, y0 = svg_point((1, 0))
    x1, y1 = svg_point((0, 1))
    return [([(x0-ARROW_LENGTH, y0-ARROW_HALF_WIDTH), (x0, y0),
              (x0-ARROW_LENGTH, y0+ARROW_HALF_WIDTH)], field(1, 0)),
            ([(x1-ARROW_HALF_WIDTH, y1+ARROW_LENGTH), (x1, y1),
              (x1+ARROW_HALF_WIDTH, y1+ARROW_LENGTH)], field(0, 1))]


def write_svg(path: Path, strokes: list[Stroke], width_mm: float,
              font_bytes: bytes, font_license: str,
              drawio_compatible: bool) -> None:
    if drawio_compatible:
        view_x, view_y, view_width, view_height = DRAWIO_VIEWBOX
        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{view_width:g}" '
            f'height="{view_height:g}" viewBox="{view_x:g} {view_y:g} '
            f'{view_width:g} {view_height:g}" '
            'preserveAspectRatio="xMidYMid meet" role="img" '
            'aria-labelledby="title desc">',
            '<title id="title">Dual-task progress space</title>',
            '<desc id="desc">Monotone, mirror-symmetric trajectories connect '
            '(0,0) to (1,1). Teal and coral progress axes define a shared '
            'continuous two-dimensional color field. White background, no '
            'ticks or grid.</desc>',
            '<defs>',
        ]
        text_font = (' font-family="Arial, Helvetica, sans-serif" '
                     'font-weight="400"')
    else:
        encoded_font = base64.b64encode(font_bytes).decode("ascii")
        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" '
            f'height="{width_mm}mm" viewBox="0 0 {CANVAS:g} {CANVAS:g}" '
            'role="img" aria-labelledby="title desc">',
            '<title id="title">Dual-task progress space</title>',
            '<desc id="desc">Monotone, mirror-symmetric trajectories connect '
            '(0,0) to (1,1). Teal and coral progress axes define a shared '
            'continuous two-dimensional color field. White background, no '
            'ticks or grid.</desc>',
            f'<metadata id="font-license">{escape(font_license)}</metadata>',
            '<defs><style>@font-face { font-family: FigureSans; src: '
            f'url("data:font/ttf;base64,{encoded_font}") format("truetype"); }} '
            'text { font-family: FigureSans, "Bitstream Vera Sans", Arial, '
            'sans-serif; font-weight: 400; }</style>',
        ]
        text_font = ''
    stroke_scale = DRAWIO_STROKE_SCALE if drawio_compatible else 1.0
    font_scale = DRAWIO_FONT_SCALE if drawio_compatible else 1.0
    for stroke in strokes:
        x1, y1 = svg_point(stroke.p0)
        x2, y2 = svg_point(stroke.p2)
        parts.append(f'<linearGradient id="gradient-{stroke.name}" '
                     f'gradientUnits="userSpaceOnUse" x1="{x1:g}" y1="{y1:g}" '
                     f'x2="{x2:g}" y2="{y2:g}" color-interpolation="sRGB">')
        for j in range(GRADIENT_STEPS+1):
            t = j/GRADIENT_STEPS
            color = svg_color(field(*stroke.point(t)))
            offset = stroke.gradient_offset(t)
            parts.append(f'<stop offset="{offset:.8f}" stop-color="{color}"/>')
        parts.append('</linearGradient>')
    parts.append('</defs>')
    if not drawio_compatible:
        parts.append('<rect width="560" height="560" fill="#fff"/>')
    parts.append('<g fill="none" stroke-linecap="round" stroke-linejoin="round">')
    for stroke in strokes:
        parts.append(f'<path id="{stroke.name}" d="{svg_path(stroke)}" '
                     f'stroke-width="{stroke.width*stroke_scale:g}" '
                     f'stroke="url(#gradient-{stroke.name})"/>')
    for points, color in arrowheads():
        commands = 'M ' + ' L '.join(point_string(p) for p in points)
        parts.append(f'<path d="{commands}" stroke="{svg_color(color)}" '
                     f'stroke-width="{AXIS_WIDTH*stroke_scale:g}"/>')
    parts.append('</g>')
    for point in [(0, 0), (1, 1)]:
        x, y = svg_point(point)
        parts.append(f'<circle cx="{x:g}" cy="{y:g}" '
                     f'r="{DOT_RADIUS*stroke_scale:g}" '
                     f'fill="{svg_color(field(*point))}"/>')
    for text, x, y, size, anchor, rotation, color in labels(drawio_compatible):
        transform = (f' transform="rotate({rotation} {x:g} {y:g})"'
                     if rotation else '')
        parts.append(f'<text x="{x:g}" y="{y:g}" '
                     f'font-size="{size*font_scale:g}" text-anchor="{anchor}" '
                     f'fill="{color}"{text_font}{transform}>'
                     f'{escape(text)}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts)+'\n', encoding='utf-8')


def gradient_stroke(pdf: canvas.Canvas, stroke: Stroke) -> None:
    """Clip a PDF axial shading to a smooth, cubic approximation of a stroke."""
    outline = pdf.beginPath()
    outline.moveTo(*stroke.outline(0, 1)[0])
    for sign in (1, -1):
        samples = [j/OUTLINE_STEPS for j in range(OUTLINE_STEPS+1)]
        if sign < 0:
            samples.reverse()
            outline.lineTo(*stroke.outline(1, sign)[0])
        for t0, t1 in zip(samples, samples[1:]):
            p0, v0 = stroke.outline(t0, sign)
            p1, v1 = stroke.outline(t1, sign)
            h = (t1-t0)/3
            outline.curveTo(p0[0]+h*v0[0], p0[1]+h*v0[1],
                            p1[0]-h*v1[0], p1[1]-h*v1[1], *p1)
    outline.close()
    pdf.saveState()
    pdf.clipPath(outline, stroke=0, fill=0)
    parameters = [j/GRADIENT_STEPS for j in range(GRADIENT_STEPS+1)]
    positions = [stroke.gradient_offset(t) for t in parameters]
    colors = [Color(*field(*stroke.point(t))) for t in parameters]
    pdf.linearGradient(*pdf_point(stroke.p0), *pdf_point(stroke.p2),
                       colors=colors, positions=positions)
    pdf.restoreState()


def write_pdf(path: Path, strokes: list[Stroke], width_mm: float,
              font_path: Path) -> None:
    width_pt = width_mm*72/25.4
    pdfmetrics.registerFont(TTFont("FigureSans", str(font_path)))
    pdf = canvas.Canvas(str(path), pagesize=(width_pt, width_pt),
                        pageCompression=1, invariant=1, initialFontName="FigureSans")
    pdf.setTitle("Dual-task progress space")
    pdf.setSubject("Symmetric trajectories in a normalized two-task progress domain")
    pdf.scale(width_pt/CANVAS, width_pt/CANVAS)
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(0, 0, CANVAS, CANVAS, fill=1, stroke=0)
    for stroke in strokes:
        gradient_stroke(pdf, stroke)
    pdf.setLineCap(1)
    pdf.setLineJoin(1)
    pdf.setLineWidth(AXIS_WIDTH)
    for points, color in arrowheads():
        pdf.setStrokeColor(Color(*color))
        arrow = pdf.beginPath()
        arrow.moveTo(points[0][0], CANVAS-points[0][1])
        for x, y in points[1:]:
            arrow.lineTo(x, CANVAS-y)
        pdf.drawPath(arrow)
    for point in [(0, 0), (1, 1)]:
        pdf.setFillColor(Color(*field(*point)))
        pdf.circle(*pdf_point(point), DOT_RADIUS, stroke=0, fill=1)
    for text, x, y, size, anchor, rotation, color in labels():
        pdf.saveState()
        pdf.translate(x, CANVAS-y)
        pdf.rotate(-rotation)
        pdf.setFont("FigureSans", size)
        pdf.setFillColor(Color(*rgb(color)))
        offset = {"start": 0, "middle": -0.5, "end": -1}[anchor]
        pdf.drawString(offset*pdfmetrics.stringWidth(text, "FigureSans", size), 0, text)
        pdf.restoreState()
    pdf.showPage()
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--width-mm', type=float, default=120.0)
    parser.add_argument('--dpi', type=int, default=400)
    args = parser.parse_args()
    if args.width_mm <= 0 or args.dpi <= 0:
        parser.error('--width-mm and --dpi must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font_path = Path(reportlab.__file__).parent/'fonts'/'Vera.ttf'
    font_license = font_path.with_name('bitstream-vera-license.txt').read_text(encoding='utf-8')
    (args.output_dir/'FONT_LICENSE.txt').write_text(font_license, encoding='utf-8')
    variants = [
        (args.output_dir/'dual_task_progress_dense', geometry()),
        (args.output_dir/'dual_task_progress_lower_right', geometry(True)),
    ]
    for stem, strokes in variants:
        write_svg(stem.with_suffix('.svg'), strokes, args.width_mm,
                  font_path.read_bytes(), font_license, False)
        drawio_path = stem.with_name(f'{stem.name}_drawio').with_suffix('.svg')
        write_svg(drawio_path, strokes, args.width_mm, b'', '', True)
        write_pdf(stem.with_suffix('.pdf'), strokes, args.width_mm, font_path)
        subprocess.run(['pdftoppm', '-png', '-singlefile', '-r', str(args.dpi),
                        str(stem.with_suffix('.pdf')), str(stem)], check=True)
        for extension in ('.svg', '.pdf', '.png'):
            print(stem.with_suffix(extension))
        print(drawio_path)


if __name__ == '__main__':
    main()
