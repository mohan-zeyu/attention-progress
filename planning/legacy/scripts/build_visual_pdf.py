#!/usr/bin/env python3
"""Export the illustrated reading path, two steps per PDF page."""
from pathlib import Path
from html import escape
import math
import re
import shutil
import subprocess
import tempfile
import textwrap
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NS = '{http://www.w3.org/2000/svg}'


def plain(text):
    text = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', text)
    return text.replace('**','').replace('*','')


def text(x,y,value,size=20,color='#253340',bold=False):
    return f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(value)}</text>'


def lines(x,y,value,width=78,size=20,color='#253340',bold=False):
    wrapped=textwrap.wrap(plain(value),width=width,break_long_words=False)
    return ''.join(text(x,y+i*(size+8),line,size,color,bold) for i,line in enumerate(wrapped))


def image(path,x,y,width):
    drawing=ET.parse(path).getroot()
    viewbox=drawing.attrib['viewBox'].split()
    height=width*float(viewbox[3])/float(viewbox[2])
    prefix=path.stem+'-'
    for element in drawing.iter():
        for key,value in list(element.attrib.items()):
            if key=='id':element.set(key,prefix+value)
            elif key=='aria-labelledby':element.set(key,' '.join(prefix+part for part in value.split()))
            elif 'url(#' in value:element.set(key,value.replace('url(#','url(#'+prefix))
    drawing.set('x',str(x));drawing.set('y',str(y))
    drawing.set('width',str(width));drawing.set('height',str(height))
    ET.register_namespace('',NS[1:-1])
    return ET.tostring(drawing,encoding='unicode'),height


def page(body,number):
    return '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1300" viewBox="0 0 900 1300"><rect width="900" height="1300" fill="white"/>'+body+text(55,1274,'Attention · visual field guide',15,'#687986')+text(815,1274,str(number),15,'#687986')+'</svg>'


def read_steps():
    chapters=[];steps=[]
    for source in sorted((ROOT/'essentials').glob('*.md')):
        content=source.read_text();title=content.splitlines()[0][2:]
        chapters.append((title,2+len(steps)//2))
        for section in re.split(r'^## ',content,flags=re.M)[1:]:
            heading,rest=section.split('\n',1)
            setup=rest.strip().split('\n\n',1)[0]
            figure=re.search(r'!\[.*?\]\(([^)]+)\)',rest).group(1)
            takeaway=re.search(r'^> (.+)$',rest,re.M).group(1)
            steps.append((title,heading,setup,(source.parent/figure).resolve(),takeaway))
    return chapters,steps


def build():
    for executable in ['rsvg-convert','pdfunite']:
        if not shutil.which(executable):raise SystemExit(f'{executable} is required')
    chapters,steps=read_steps()
    cover=text(55,90,'Attention,',48,bold=True)+text(55,149,'one change at a time.',48,bold=True)
    cover+=text(55,202,'38 pictures. 11 chapters. Follow one operation at a time.',21,'#188366')
    art,_=image(ROOT/'figures/visual/00-organizations.svg',55,249,790);cover+=art
    for i,(label,pageno) in enumerate(chapters):
        cover+=text(70,601+i*43,label,22)+text(805,601+i*43,str(pageno),20,'#188366')
    cover+=lines(55,1130,'A visual companion to the chaptered HTML book. Exact equations, supporting papers and technical qualifications remain in its reference notes.',78,18,'#687986')
    cover+=text(55,1210,'Literature checked through 10 October 2026.',17,'#687986')
    pages=[page(cover,1)]
    for batch in range(math.ceil(len(steps)/2)):
        body=''
        for position,(chapter,heading,setup,figure,takeaway) in enumerate(steps[batch*2:batch*2+2]):
            top=44+position*597
            body+=text(55,top,chapter,16,'#188366',True)
            heading_lines=textwrap.wrap(heading,width=58,break_long_words=False)
            body+=''.join(text(55,top+42+i*32,line,26,bold=True) for i,line in enumerate(heading_lines))
            setup_y=top+78+(len(heading_lines)-1)*32
            body+=lines(55,setup_y,setup,82,19)
            setup_height=len(textwrap.wrap(plain(setup),width=82,break_long_words=False))*27
            image_y=setup_y+setup_height+8
            art,art_height=image(figure,55,image_y,790);body+=art
            takeaway_y=image_y+art_height+16
            take_lines=textwrap.wrap(plain(takeaway),width=83,break_long_words=False)
            box_height=18+len(take_lines)*25
            assert takeaway_y+box_height<top+580,(figure.name,takeaway_y+box_height,top+580)
            body+=f'<rect x="55" y="{takeaway_y}" width="790" height="{box_height}" fill="#f4faf7"/><rect x="55" y="{takeaway_y}" width="3" height="{box_height}" fill="#209f78"/>'
            body+=''.join(text(70,takeaway_y+25+i*25,line,18,'#188366',True) for i,line in enumerate(take_lines))
            if position==0:body+='<line x1="55" y1="618" x2="845" y2="618" stroke="#c1dfd1"/>'
        pages.append(page(body,batch+2))
    destination=ROOT/'attention-visual-guide.pdf'
    with tempfile.TemporaryDirectory(prefix='attention-pdf-') as tmp:
        outputs=[]
        for i,svg in enumerate(pages):
            source=Path(tmp)/f'{i:02}.svg';target=source.with_suffix('.pdf')
            source.write_text(svg)
            subprocess.run(['rsvg-convert','-f','pdf','-o',str(target),str(source)],check=True,capture_output=True)
            outputs.append(str(target))
        subprocess.run(['pdfunite',*outputs,str(destination)],check=True,capture_output=True)
    print(f'Exported {len(pages)} pages: {destination.name}')


if __name__=='__main__':build()
