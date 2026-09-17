from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io, json, subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/shuyuan/Downloads/drawer-output000-original-source003-top (1).mp4')
OUTPUT = Path('/Users/shuyuan/Downloads/drawer-no-stop-review-156-20260915/videos/episode_000.mp4')

def frame(video, time, crop=None):
    raw = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', str(time), '-i', str(video), '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'], check=True, capture_output=True).stdout
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    return img.crop(crop) if crop else img

def main():
    actions = [('approach', 8.8, (60,0,540,480)), ('pick', 11.25, (60,0,540,480)), ('grasp_drawer', 5.3, (320,0,800,480)), ('open', 7.8, (320,0,800,480)), ('place', 14.3, (180,0,660,480)), ('close', 20.0, (320,0,800,480))]
    for name, time, crop in actions:
        frame(SOURCE, time).save(ROOT/'assets'/f'{name}_full.png')
        frame(SOURCE, time, crop).save(ROOT/'assets'/f'{name}.png')
    for i, time in enumerate([0.3, 4.8, 7.8, 10.0, 13.2, 16.0]):
        frame(OUTPUT, time, (124, 0, 486, 362)).save(ROOT/'assets'/f'ctr_{i}.png')
    allitems = [(name, time, ROOT/'assets'/f'{name}.png') for name,time,crop in actions]
    allitems += [(f'ctr_{i}', t, ROOT/'assets'/f'ctr_{i}.png') for i,t in enumerate([0.3,4.8,7.8,10.0,13.2,16.0])]
    sheet = Image.new('RGB', (6*280,2*310),'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
    for i,(name,time,path) in enumerate(allitems):
        x,y=(i%6)*280,(i//6)*310
        sheet.paste(Image.open(path).resize((280,280)),(x,y))
        draw.text((x+6,y+286),f'{name} | {time:.1f}s',font=font,fill='black')
    sheet.save(ROOT/'work'/'selected_frames.png')
    (ROOT/'assets'/'frames.json').write_text(json.dumps({'source_video':str(SOURCE),'ctr_video':str(OUTPUT),'source_actions':[{'action':name,'time_s':time,'crop_xyxy':crop} for name,time,crop in actions],'source_fps':30,'note':'Action listing follows the requested schematic order; original chronology is grasp/open before approach/pick.'},indent=2))

if __name__ == '__main__':
    main()
