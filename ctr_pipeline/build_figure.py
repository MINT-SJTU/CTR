"""Build the paper figure as matching editable SVG and vector PDF.

Photographs are direct crops. The six manually traced foreground overlays are
explicitly illustrative, not claimed as outputs of a segmentation model.
Joint-progress geometry and OKLab interpolation reuse the supplied function.
"""
from pathlib import Path
import base64
import importlib.util
import json
import math
import sys
import xml.etree.ElementTree as ET
from html import escape
from reportlab.pdfgen import canvas
from reportlab.lib.colors import toColor as HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.svgpath import SvgPath

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
W, H = 1800, 675
WIDTH_MM = 180
INK, MUTED, LINE = '#25333F', '#6F7B88', '#B1BAC5'
PICK, OPEN, PLACE, CLOSE = '#CC8068', '#4F9E98', '#648BB7', '#977CB0'
PURPLE, PALE, PROTECTED = '#8870B5', '#F5F1FA', '#148F83'
ORDER = ['approach', 'pick', 'grasp_drawer', 'open', 'place', 'close']
COLORS = [PICK, PICK, OPEN, OPEN, PLACE, CLOSE]
FONT_DIR = Path('/System/Library/Fonts/Supplemental')
pdfmetrics.registerFont(TTFont('Arial', str(FONT_DIR/'Arial.ttf')))
pdfmetrics.registerFont(TTFont('ArialBold', str(FONT_DIR/'Arial Bold.ttf')))

spec = importlib.util.spec_from_file_location('joint_progress', ASSETS/'joint_progress.py')
progress = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = progress
spec.loader.exec_module(progress)
progress.PALETTE = {'origin':'#BEC8D1','task1':PICK,'task2':OPEN,'joint':PLACE}
progress.LAB_CORNERS = tuple(progress.to_oklab(progress.rgb(progress.PALETTE[k])) for k in ('origin','task1','task2','joint'))

class Figure:
    def __init__(self):
        s = WIDTH_MM*72/25.4/W
        self.pdf = canvas.Canvas(str(ROOT/'ctr_pipeline.pdf'), pagesize=(W*s,H*s),pageCompression=1,invariant=1)
        self.pdf.setTitle('CTR data processing pipeline')
        self.pdf.setSubject('Temporal annotation, foreground segmentation and constraint-preserving temporal recombination')
        self.pdf.scale(s,s)
        self.pdf.translate(0,H)
        self.pdf.scale(1,-1)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="{WIDTH_MM}mm" height="{WIDTH_MM*H/W}mm" viewBox="0 0 {W} {H}">', '<title>CTR data processing pipeline</title>', '<desc>Six representative source-video crops are shown in a schematic task order. Foreground overlays are illustrative manual traces. The three recombined schedules and joint-progress curves are schematic illustrations, not measured video outputs or empirical densities.</desc>']
        self.rect(0,0,W,H,'white',None)

    def group(self, name, ty=0):
        self.pdf.saveState()
        self.pdf.translate(0,ty)
        self.svg.append(f'<g id="{name}" transform="translate(0 {ty})">')

    def end(self):
        self.svg.append('</g>')
        self.pdf.restoreState()

    def color(self, stroke=None, fill=None, width=2, alpha=1):
        if stroke:
            self.pdf.setStrokeColor(HexColor(stroke))
        if fill:
            self.pdf.setFillColor(HexColor(fill))
        self.pdf.setLineWidth(width)
        self.pdf.setLineCap(1)
        self.pdf.setLineJoin(1)
        self.pdf.setFillAlpha(alpha)
        self.pdf.setStrokeAlpha(alpha)

    def rect(self,x,y,w,h,fill='white',stroke=LINE,r=0,lw=1.5,alpha=1):
        self.pdf.saveState(); self.color(stroke,fill,lw,alpha)
        if r:
            self.pdf.roundRect(x,y,w,h,r,stroke=bool(stroke),fill=bool(fill))
        else:
            self.pdf.rect(x,y,w,h,stroke=bool(stroke),fill=bool(fill))
        self.pdf.restoreState()
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{lw}" opacity="{alpha}"/>')

    def circle(self,x,y,r,fill=INK,stroke=None,lw=2):
        self.pdf.saveState(); self.color(stroke,fill,lw)
        self.pdf.circle(x,y,r,stroke=bool(stroke),fill=bool(fill));self.pdf.restoreState()
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{lw}"/>')

    def path(self,d,stroke=LINE,fill=None,lw=2,alpha=1,dash=None):
        obj=SvgPath(d)
        p=self.pdf.beginPath(); j=0
        for op in obj.operators:
            n=(2,2,6,0)[op]
            coords=obj.points[j:j+n]; j+=n
            if op==0:p.moveTo(*coords)
            elif op==1:p.lineTo(*coords)
            elif op==2:p.curveTo(*coords)
            else:p.close()
        self.pdf.saveState(); self.color(stroke,fill,lw,alpha)
        if dash:self.pdf.setDash(dash)
        self.pdf.drawPath(p,stroke=bool(stroke),fill=bool(fill),fillMode=0)
        self.pdf.restoreState()
        dashattr=f' stroke-dasharray="{",".join(map(str,dash))}"' if dash else ''
        self.svg.append(f'<path d="{d}" stroke="{stroke or "none"}" fill="{fill or "none"}" stroke-width="{lw}" stroke-linecap="round" stroke-linejoin="round" opacity="{alpha}"{dashattr}/>')

    def text(self,x,y,txt,size=24,color=INK,bold=False,anchor='start',rotate=0):
        self.pdf.saveState();self.pdf.translate(x,y);self.pdf.rotate(rotate);self.pdf.scale(1,-1)
        self.pdf.setFillColor(HexColor(color));self.pdf.setFont('ArialBold' if bold else 'Arial',size)
        fn={'start':self.pdf.drawString,'middle':self.pdf.drawCentredString,'end':self.pdf.drawRightString}[anchor]
        fn(0,0,txt);self.pdf.restoreState()
        transform=f' transform="rotate({rotate} {x} {y})"' if rotate else ''
        self.svg.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}" text-anchor="{anchor}"{transform}>{escape(txt)}</text>')

    def image(self,name,x,y,w,h=None):
        h=w if h is None else h
        file=ASSETS/f'{name}.png'
        self.pdf.saveState();self.pdf.translate(x,y+h);self.pdf.scale(1,-1)
        self.pdf.drawImage(str(file),0,0,width=w,height=h,mask='auto');self.pdf.restoreState()
        encoded=base64.b64encode(file.read_bytes()).decode()
        self.svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/png;base64,{encoded}"/>')

    def arrow(self,points,color=LINE,lw=2.6,dash=None,head=8):
        self.path('M '+' L '.join(f'{x},{y}' for x,y in points),color,lw=lw,dash=dash)
        x,y=points[-1];x0,y0=points[-2];theta=math.atan2(y-y0,x-x0)
        a=(x-head*math.cos(theta)+head*.5*math.sin(theta),y-head*math.sin(theta)-head*.5*math.cos(theta))
        b=(x-head*math.cos(theta)-head*.5*math.sin(theta),y-head*math.sin(theta)+head*.5*math.cos(theta))
        self.path(f'M {x},{y} L {a[0]},{a[1]} L {b[0]},{b[1]} Z',None,color)

    def title(self,x,y,letter,txt,w):
        self.text(x,y,f'({letter})',28,MUTED)
        self.text(x+47,y,txt,29,INK,True)
        self.path(f'M {x},{y+18} H {x+w}', '#DEE4E9',lw=1.4)

    def save(self):
        self.pdf.showPage();self.pdf.save()
        (ROOT/'ctr_pipeline.svg').write_text('\n'.join(self.svg+['</svg>']))

F=Figure()

def lock(x,y,s=1,color=PROTECTED):
    F.rect(x-12*s,y-1*s,24*s,21*s,None,color,3*s,2.2*s)
    F.path(f'M {x-7*s},{y-1*s} V {y-9*s} C {x-7*s},{y-18*s} {x+7*s},{y-18*s} {x+7*s},{y-9*s} V {y-1*s}',color,lw=2.2*s)
    F.circle(x,y+7*s,2*s,color)
    F.path(f'M {x},{y+8*s} V {y+13*s}',color,lw=1.7*s)

def shuffle(x,y,s=1,color=PURPLE):
    F.path(f'M {x-17*s},{y-11*s} H {x-10*s} C {x-1*s},{y-11*s} {x+1*s},{y+11*s} {x+10*s},{y+11*s} H {x+18*s}',color,lw=2.8*s)
    F.path(f'M {x-17*s},{y+11*s} H {x-10*s} C {x-1*s},{y+11*s} {x+1*s},{y-11*s} {x+10*s},{y-11*s} H {x+18*s}',color,lw=2.8*s)
    F.path(f'M {x+12*s},{y-17*s} L {x+18*s},{y-11*s} L {x+12*s},{y-5*s} M {x+12*s},{y+5*s} L {x+18*s},{y+11*s} L {x+12*s},{y+17*s}',color,lw=2.8*s)

def openai(x,y,size):
    d=ET.parse(ASSETS/'openai.svg').getroot().find('{http://www.w3.org/2000/svg}path').attrib['d']
    obj=SvgPath(d); j=0;parts=[];scale=size/24
    for op in obj.operators:
        n=(2,2,6,0)[op];v=obj.points[j:j+n];j+=n
        coords=[v[k]*scale+(x if k%2==0 else y) for k in range(n)]
        parts.append(('M','L','C','Z')[op]+' '+','.join(map(str,coords)))
    F.path(' '.join(parts),None,INK)

def reviewer(x,y):
    F.circle(x,y,11,None,MUTED,2.5)
    F.path(f'M {x-22},{y+36} V {y+29} C {x-22},{y+13} {x+22},{y+13} {x+22},{y+29} V {y+36}',MUTED,lw=2.5)
    F.circle(x+24,y+28,12,'white',PROTECTED,2)
    F.path(f'M {x+18},{y+28} L {x+22},{y+32} L {x+30},{y+23}',PROTECTED,lw=2.3)

def code_icon(x,y):
    F.rect(x,y,101,92,'#FAFBFC',INK,9,2.1)
    F.path(f'M {x},{y+19} H {x+101}',LINE,lw=1.5)
    for off,col in [(12,PICK),(23,OPEN),(34,CLOSE)]:F.circle(x+off,y+10,2.2,col)
    F.path(f'M {x+27},{y+34} L {x+15},{y+47} L {x+27},{y+59} M {x+75},{y+34} L {x+87},{y+47} L {x+75},{y+59} M {x+57},{y+32} L {x+45},{y+61}',INK,lw=3)
    for dx in [18,46,74]: F.rect(x+dx,y+73,11,8,'#DFD5EB',PURPLE,1,1)
    F.path(f'M {x+29},{y+77} H {x+46} M {x+57},{y+77} H {x+74}',PURPLE,lw=1.5)

# Polygon contours are designed for the exact source-frame crop coordinates.
# Kept as vectors to separate illustrative overlays from the unchanged images.
MASKS = {
 'approach':[
  (PICK,[(0,49),(73,64),(97,100),(145,103),(181,145),(184,186),(230,209),(232,230),(199,244),(155,233),(110,212),(52,181),(0,155)]),
  (PICK,[(218,335),(242,332),(256,343),(249,372),(225,373),(215,359)])],
 'pick':[
  (PICK,[(0,42),(62,53),(104,51),(144,74),(150,121),(183,142),(207,183),(216,206),(264,227),(264,255),(230,267),(180,257),(162,234),(119,216),(90,180),(47,164),(0,141)]),
  (PICK,[(224,263),(246,261),(252,281),(238,294),(217,287)])],
 'grasp_drawer':[
  (OPEN,[(479,240),(440,224),(408,219),(396,183),(364,184),(339,204),(305,210),(282,235),(250,230),(233,259),(203,270),(211,296),(253,292),(275,320),(296,326),(317,350),(344,353),(371,375),(402,380),(423,411),(479,431)]),
  (OPEN,[(24,182),(81,260),(221,240),(278,214),(279,246),(88,294),(28,220)])],
 'open':[
  (OPEN,[(480,329),(435,313),(410,284),(366,279),(357,265),(327,269),(300,288),(290,313),(256,324),(226,349),(218,376),(251,389),(266,413),(296,425),(326,429),(346,455),(384,461),(417,480),(480,480)]),
  (OPEN,[(101,204),(246,174),(331,285),(217,334),(167,324),(135,271),(82,247)])],
 'place':[
  (PICK,[(0,65),(200,9),(247,0),(365,0),(375,32),(399,71),(393,137),(415,176),(425,223),(401,266),(355,290),(326,263),(313,224),(280,194),(286,167),(263,135),(265,100),(192,138),(0,228)]),
  (PICK,[(313,313),(341,309),(358,326),(345,348),(317,346),(306,332)]),
  (OPEN,[(285,229),(368,187),(432,293),(366,366),(292,351),(268,281)]),
  (OPEN,[(480,294),(451,310),(438,346),(441,380),(417,409),(445,443),(480,458)])],
 'close':[
  (OPEN,[(480,278),(452,264),(420,246),(399,231),(372,234),(350,219),(325,225),(291,255),(278,267),(243,274),(221,289),(232,318),(259,320),(278,343),(306,352),(324,383),(351,393),(380,413),(407,437),(449,445),(480,464)]),
  (OPEN,[(71,183),(216,153),(276,227),(190,260),(130,259),(105,226)])]
}

def masks(name,x,y,size):
    for color,pts in MASKS[name]:
        d='M '+' L '.join(f'{x+px*size/480},{y+py*size/480}' for px,py in pts)+' Z'
        F.path(d,None,color,alpha=.34)
        F.path(d,color,lw=1.25)

def filmstrip(x,y,size,gap=7,segmented=False,masked=False,stack=False,labels=True):
    width=6*size+5*gap
    if stack:
        # Repeat the exact same dataset card, including all six images.
        for offset in [18,9,0]:
            F.group(f'dataset-copy-{offset}')
            F.rect(x-6+offset,y-34-offset,width+12,size+45,'white','#DCE2E7',5,1.5)
            filmstrip(x+offset,y-offset,size,gap,segmented,masked,False,False)
            F.end()
        if labels:
            F.text(x,y-69,'Unsegmented video',23,MUTED)
            F.text(x+width,y-69,'time',21,MUTED,anchor='end')
        return width
    if segmented:
        span=2*size+gap
        # Three two-frame groups; the protected group retains its two teaser colors.
        for k,col in [(0,PICK),(1,OPEN)]:
            F.rect(x+k*2*(size+gap),y-23,span,12,col,None)
        last=x+4*(size+gap)
        F.rect(last,y-23,span,12,'#DCD4E8',None)
        F.rect(last,y-23,size+gap/2,12,PLACE,None)
        F.rect(last+size+gap/2,y-23,size+gap/2,12,CLOSE,None)
        F.path(f'M {last},{y-31} V {y-36} H {last+span} V {y-31}',CLOSE,lw=1.5)
    else:
        F.rect(x,y-23,width,12,'#BBC3CC',None)
    for i,name in enumerate(ORDER):
        px=x+i*(size+gap)
        F.image(name,px,y,size)
        if segmented:F.rect(px,y,size,size,None,COLORS[i],0,2.4)
        if masked:masks(name,px,y,size)
    if labels:
        if segmented:
            for i,txt,col in [(0,'Pick up cube',PICK),(1,'Open drawer',OPEN),(2,'Place → close',CLOSE)]:
                F.text(x+(2*i+1)*size+(2*i+.5)*gap,y-48,txt,22,col,True,'middle')
        else:
            F.text(x,y-69,'Unsegmented video',23,MUTED)
            F.text(x+width,y-69,'time',21,MUTED,anchor='end')
    return width

def joint_plot(x,y,side,protected=False):
    # Original function's 13 monotone trajectories in the same OKLab field.
    saved_corners=progress.LAB_CORNERS
    if protected:
        progress.LAB_CORNERS=tuple(progress.to_oklab(progress.rgb(c)) for c in (PLACE,PLACE,CLOSE,CLOSE))
    for stroke in progress.geometry(single_lower_right=protected):
        pts=[stroke.point(i/64) for i in range(65)]
        for a,b in zip(pts,pts[1:]):
            mid=((a[0]+b[0])/2,(a[1]+b[1])/2)
            color='#'+''.join(f'{round(c*255):02x}' for c in progress.field(*mid))
            F.path(f'M {x+a[0]*side},{y+(1-a[1])*side} L {x+b[0]*side},{y+(1-b[1])*side}',color,lw=(4 if protected else 3.1) if stroke.kind=='trajectory' else 2)
    cx,cy=(PLACE,CLOSE) if protected else (PICK,OPEN)
    F.arrow([(x+side-9,y+side),(x+side+5,y+side)],cx,lw=2,head=7)
    F.arrow([(x,y+9),(x,y-5)],cy,lw=2,head=7)
    F.circle(x,y+side,3.7,PLACE if protected else '#BEC8D1')
    F.circle(x+side,y,3.7,CLOSE if protected else PLACE)
    F.text(x+side/2,y+side+33,'Place' if protected else 'Pick up cube',21 if protected else 22,cx,True,'middle')
    F.text(x-29,y+side/2,'Close' if protected else 'Open drawer',21 if protected else 19,cy,True,'middle',-90)
    for dx,dy,txt,anchor in [(-7,side+20,'(0, 0)','end'),(-7,3,'(0, 1)','end'),(side+9,side+20,'(1, 0)','start'),(side+9,3,'(1, 1)','start')]:
        F.text(x+dx,y+dy,txt,16,'#9299A6',anchor=anchor)
    if protected:
        lock(x+side*.42,y+side*.52,.7,PROTECTED)
    progress.LAB_CORNERS=saved_corners

def photo_pair(x,y,size,names,colors,gap=5):
    for i,(name,col) in enumerate(zip(names,colors)):
        px=x+i*(size+gap)
        F.image(name,px,y,size);F.rect(px,y,size,size,None,col,0,2.4)


# A — copied source-video cards.
F.group('a-input-dataset')
F.title(30,47,'a','Raw demonstration dataset',685)
filmstrip(42,157,100,7,stack=True)
F.text(42,289,'Representative keyframes · schematic task order',21,MUTED)
F.end()

# The dataset enters the reviewed code along an orthogonal route.
F.group('annotation-code-generation-and-review')
openai(751,172,74)
F.text(788,280,'GPT6 Astra',24,INK,True,'middle')
F.arrow([(716,205),(738,205)],INK,head=6)
code_icon(922,165)
F.text(973,280,'Video pipeline',23,INK,True,'middle')
F.text(973,306,'Generated code',21,MUTED,anchor='middle')
F.arrow([(838,205),(905,205)],INK)
F.text(873,183,'generate',18,MUTED,anchor='middle')
reviewer(973,68)
F.text(973,36,'Human review',23,MUTED,anchor='middle')
F.arrow([(973,111),(973,155)],PROTECTED,lw=2.3)
F.arrow([(706,116),(939,116),(939,155)],LINE,lw=2,dash=[6,5])
F.text(810,103,'dataset',19,MUTED,anchor='middle')
F.arrow([(1035,205),(1132,205)],INK)
F.text(1083,185,'run',20,MUTED,anchor='middle')
F.end()

# B — six colored keyframes in three groups.
F.group('b-temporal-segmentation')
F.title(1150,47,'b','Temporal segmentation',620)
filmstrip(1150,145,96,7,segmented=True)
F.text(1150,289,'Reschedulable',22,MUTED)
F.path('M 1150,262 V 268 H 1555 V 262',LINE,lw=1.5)
lock(1604,269,.58,CLOSE)
F.text(1621,289,'Protected',22,CLOSE)
F.end()

# RobotSeg shares the foreground-segmentation heading's baseline.
F.group('robotseg-foreground-extraction')
F.arrow([(1670,301),(1670,316)],INK,head=5)
F.rect(1568,324,202,42,'#F6F8FA','#DEE4E9',7,1.5)
F.rect(1580,330,35,30,None,INK,3,1.4)
F.path('M 1585,353 L 1592,348 L 1591,339 L 1598,334 L 1604,339 L 1602,347 L 1611,352',OPEN,lw=4.7)
F.circle(1597,340,2.5,PICK)
F.text(1628,354,'RobotSeg',24,INK,True)
F.arrow([(1670,378),(1670,395)],INK,head=6)
F.end()

# C — the two explanatory lines have been removed; the mask legend remains.
F.group('c-foreground-masks')
F.title(1150,354,'c','Foreground segmentation',620)
filmstrip(1150,463,96,7,segmented=True,masked=True)
F.rect(1150,586,15,15,PICK,None);F.text(1175,601,'Left arm + cube',22,MUTED)
F.rect(1440,586,15,15,OPEN,None);F.text(1465,601,'Right arm + drawer',22,MUTED)
F.end()

# The same teaser module with reduced vertical spacing.
F.group('ctr-teaser-module',-70)
F.rect(992,476,124,253,PALE,'#C9BADE',12,1.5)
shuffle(1054,509,.95)
F.text(1054,547,'Recombine',20,PURPLE,True,'middle')
F.path('M 1012,562 H 1096','#DFD5EB',lw=1.2)
F.text(1054,602,'CTR',34,PURPLE,True,'middle')
F.text(1054,630,'Keep local',19,MUTED,anchor='middle')
F.text(1054,654,'arm motion',19,MUTED,anchor='middle')
F.path('M 1012,668 H 1096','#DFD5EB',lw=1.2)
lock(1054,689,.7)
F.text(1054,720,'Preserve',20,PROTECTED,True,'middle')
F.arrow([(1138,582),(1123,582)],PURPLE,lw=2.7,head=6)
F.arrow([(984,582),(969,582)],PURPLE,lw=2.7,head=6)
F.end()

def schedule_sample(y,pick_offset,open_offset,index):
    """A thumbnail row above two adjacent task-clock bands; no scaling of motion."""
    x=310;length=104
    F.group(f'recombined-schedule-{index}')
    pairs=sorted([(pick_offset,ORDER[:2],COLORS[:2]),
                  (open_offset,ORDER[2:4],COLORS[2:4])],key=lambda pair:pair[0])
    for i,(_,names,colors) in enumerate(pairs):
        photo_pair(x+i*74,y,35,names,colors,gap=2)
    py=y+46;oy=py+10
    F.rect(x+pick_offset,py,length,8,PICK,None)
    F.rect(x+open_offset,oy,length,8,OPEN,None)
    endpoint=x+max(pick_offset,open_offset)+length
    F.path(f'M {endpoint+6},{oy+4} H 507',LINE,lw=1.6)
    F.end()

# D — progress support, three actual timeline schematics, and a protected tail.
F.group('d-recombined-distribution')
F.title(30,354,'d','Constraint-preserving recombination',940)
F.text(175,412,'Joint progress',24,INK,True,'middle')
F.group('joint-progress-reschedulable')
joint_plot(112,471,126)
F.end()
F.text(175,652,'Multiple admissible schedules',17,MUTED,anchor='middle')

F.text(383,412,'Sampled schedules',24,INK,True,'middle')
schedule_sample(437,0,40,1)
schedule_sample(509,0,6,2)
schedule_sample(581,40,0,3)
F.path('M 507,497 V 641',LINE,lw=1.6)
F.arrow([(507,561),(545,561)],LINE,lw=2,head=6)

lock(573,488,.65,CLOSE)
F.text(593,496,'Fixed tail',23,CLOSE,True)
photo_pair(554,526,71,ORDER[4:],COLORS[4:])
F.rect(554,607,71,10,PLACE,None)
F.rect(630,607,71,10,CLOSE,None)
F.text(589,642,'Place',20,PLACE,True,'middle')
F.text(665,642,'Close',20,CLOSE,True,'middle')
F.arrow([(622,518),(634,518)],MUTED,lw=1.4,head=4)

F.text(863,412,'Order preserved',23,INK,True,'middle')
F.group('joint-progress-protected')
joint_plot(800,471,126,protected=True)
F.end()
F.end()

F.save()
(ROOT/'figure_metadata.json').write_text(json.dumps({
    'width_mm':WIDTH_MM,'height_mm':WIDTH_MM*H/W,
    'colors':{'pick':PICK,'open':OPEN,'place':PLACE,'close':CLOSE,'ctr':PURPLE},
    'foreground_masks':'Illustrative manually traced vector overlays, not RobotSeg outputs; disclosed here and in README, explanatory caption removed at user request.',
    'source_keyframe_order':'Requested schematic task order, differs from source chronology.',
    'ctr_examples':'The recomposed-video filmstrip was removed at user request; its extracted source assets remain available.',
    'joint_progress':'13 monotone trajectories and OKLab field from user-supplied generate_figure.py; schematic, not empirical density.',
    'protected_joint_progress':'One lower-right place-before-close trajectory, in the teaser blue/purple palette.',
    'schedule_samples':[
      {'sample':1,'pick_start':0,'open_start':40,'duration':104,'photo_order':['pick','open']},
      {'sample':2,'pick_start':0,'open_start':6,'duration':104,'photo_order':['pick','open']},
      {'sample':3,'pick_start':40,'open_start':0,'duration':104,'photo_order':['open','pick']}],
    'joint_plot_side':126,
    'dataset_stack':'Three identical copies of the complete six-frame card, translated only.'
},indent=2))
print(ROOT/'ctr_pipeline.pdf')
