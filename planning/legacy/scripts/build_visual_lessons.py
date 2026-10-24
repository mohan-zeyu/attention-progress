#!/usr/bin/env python3
"""Draw the visual reading path. Every picture explains one operation."""
from pathlib import Path
from html import escape
import math
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures' / 'visual'
COL = {'ink':'#253340','muted':'#687986','line':'#cad5db','blue':'#4287b9','green':'#209f78','gold':'#d49a30','red':'#cf6c66','purple':'#9272b7','paper':'#f3f6f7','white':'#ffffff'}
INVENTORY = []


class Drawing:
    def __init__(self, name, desc, height=300):
        self.name, self.desc, self.height = name, desc, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 {height}" role="img" aria-labelledby="title desc">', f'<title id="title">{escape(name)}</title><desc id="desc">{escape(desc)}</desc>', '<defs>']
        for key in ('ink','muted','blue','green','gold','red','purple'):
            self.parts.append(f'<marker id="a-{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{COL[key]}"/></marker>')
        self.parts += ['</defs>', f'<rect width="900" height="{height}" fill="white"/>']

    def text(self,x,y,s,size=22,color='ink',bold=False,anchor='middle'):
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{COL.get(color,color)}">{escape(str(s))}</text>')

    def box(self,x,y,w,h,label='',color='blue',fill=False,size=21,opacity=1):
        c=COL.get(color,color)
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{c if fill else COL["paper"]}" fill-opacity="{opacity}" stroke="{c}" stroke-width="1.7"/>')
        if label:self.text(x+w/2,y+h/2+size*.35,label,size,'white' if fill else ('muted' if color=='line' else color))

    def circle(self,x,y,r,label='',color='blue',fill=False,size=20):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{COL[color] if fill else COL["white"]}" stroke="{COL[color]}" stroke-width="2"/>')
        if label:self.text(x,y+size*.35,label,size,'white' if fill else color)

    def line(self,x1,y1,x2,y2,color='line',width=2,dash=False,arrow=False):
        extra=f' marker-end="url(#a-{color})"' if arrow else ''
        if dash:extra+=' stroke-dasharray="5 5"'
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COL[color]}" stroke-width="{width}"{extra}/>')

    def arrow(self,x1,y1,x2,y2,color='muted',dash=False):
        self.line(x1,y1,x2,y2,color,2,dash,True)

    def path(self,points,color='muted',arrow=False):
        d='M'+'L'.join(f'{x} {y}' for x,y in points)
        extra=f' marker-end="url(#a-{color})"' if arrow else ''
        self.parts.append(f'<path d="{d}" fill="none" stroke="{COL[color]}" stroke-width="2"{extra}/>')

    def grid(self,x,y,rows=4,cols=4,cell=25,color='blue',values=None):
        for r in range(rows):
            for c in range(cols):
                v=values[r][c] if values is not None else .25+.65*((r*7+c*3)%9)/8
                self.parts.append(f'<rect x="{x+c*cell}" y="{y+r*cell}" width="{cell-2}" height="{cell-2}" rx="2" fill="{COL[color]}" fill-opacity="{max(.06,v)}"/>')

    def records(self,x,y,n=6,w=57,h=54,gap=12,selected=None,labels=None,color='blue'):
        for i in range(n):
            tone=color if selected is None else ('green' if i in selected else 'line')
            self.box(x+i*(w+gap),y,w,h,labels[i] if labels else '',tone)
            if not labels:
                for j in range(3):self.line(x+i*(w+gap)+10,y+15+j*11,x+i*(w+gap)+w-10,y+15+j*11,tone)

    def divider(self,x):self.line(x,32,x,self.height-25)

    def save(self):
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/(self.name+'.svg')).write_text('\n'.join(self.parts+['</svg>\n']))
        INVENTORY.append({'file':self.name+'.svg','description':self.desc})


def overview():
    d=Drawing('00-query','Illustrative read: a query about delivery uses the Friday correction among stored conversation records.',275)
    d.text(30,38,'Earlier conversation',21,'muted',anchor='start')
    d.records(35,115,6,118,58,19,[3],['7314','Tuesday','…','Friday','…','…'])
    d.box(517,20,220,52,'Delivery day?','gold')
    d.arrow(625,72,505,111,'green')
    d.arrow(505,175,625,223,'green')
    d.text(710,241,'Friday',28,'green',True)
    d.save()

    d=Drawing('00-organizations','History can remain separate records or update a bounded aggregate state. At fixed model dimensions, record count grows but decode-state dimensions stay fixed.',310)
    d.text(235,35,'Keep records',25,'blue',True);d.text(685,35,'Update a state',25,'purple',True);d.divider(450)
    d.records(55,80,5,57,50,14)
    d.text(234,160,'More history',21,'muted');d.arrow(235,169,235,197)
    d.records(38,220,7,43,45,12)
    d.text(564,84,'Before',20,'muted');d.text(804,84,'After',20,'muted')
    d.grid(520,104,3,4,22,'purple')
    d.arrow(613,137,633,137,'purple');d.box(639,116,90,43,'Update','purple',size=18)
    d.arrow(733,137,754,137,'purple');d.grid(760,104,3,4,22,'purple',[[.2,.8,.4,.7],[.7,.3,.9,.5],[.4,.9,.2,.6]])
    d.box(639,210,90,43,'New token','gold',size=17);d.arrow(684,205,684,164,'gold')
    d.text(685,294,'Same dimensions',21,'purple')
    d.save()

    d=Drawing('00-choices','Three operations within addressable records: sharing heads, shrinking each record and pooling positions. These are operations, with named designs shown underneath.',290)
    for x,label,name in [(150,'Share heads','MQA / GQA'),(450,'Shrink each record','MLA'),(750,'Pool positions','CSA / HCA')]:
        d.text(x,36,label,23,bold=True);d.text(x,270,name,22,'green',True)
    for i in range(4):
        d.circle(75+i*50,92,13,color='gold');d.arrow(75+i*50,108,150,158,'blue')
    d.box(117,167,66,52,'KV')
    d.box(360,76,180,46,color='blue');d.arrow(450,126,450,161,'purple');d.box(414,172,72,43,'c','purple')
    d.records(627,78,4,45,42,16)
    for i in range(4):d.arrow(650+i*61,123,750,164,'green')
    d.box(716,174,68,43,'c','green')
    d.save()


def foundations():
    d=Drawing('01-bottleneck','One summary is a bottleneck. Keeping separate contextual records allows different output steps to consult different positions.',285)
    d.text(225,35,'One summary',24,'muted',True);d.text(675,35,'Several accessible records',24,'green',True);d.divider(450)
    d.records(42,77,4,66,42,29)
    for i in range(4):d.arrow(75+i*95,121,225,169)
    d.box(195,172,60,44,'S','purple');d.arrow(225,219,225,251)
    d.text(225,276,'Every output uses S',21,'muted')
    d.records(492,77,4,66,42,29)
    d.circle(595,233,25,'q₁','gold');d.circle(755,233,25,'q₂','gold')
    d.arrow(595,205,525,125,'green');d.arrow(755,205,810,125,'green')
    d.save()

    d=Drawing('01-qkv','A query matches keys. Softmax supplies weights, shown illustratively as 0.1, 0.2, 0.6 and 0.1. The output combines the corresponding values.',315)
    d.circle(70,152,32,'Q','gold')
    for i,w in enumerate([.1,.2,.6,.1]):
        y=33+i*68
        d.arrow(104,152,227,y+20,'gold');d.box(235,y,66,40,'K'+str(i+1))
        d.box(345,y+7,210*w,26,color='green',fill=True)
        d.text(590,y+28,str(w),20,'green')
        d.box(650,y,66,40,'V'+str(i+1),'purple')
        d.arrow(721,y+20,807,152,'purple')
    d.circle(837,152,29,'Σ','green')
    d.text(70,294,'Query',20,'gold');d.text(268,294,'Match keys',20,'blue');d.text(465,294,'Weights',20,'green');d.text(705,294,'Combine values',20,'purple')
    d.save()

    d=Drawing('01-heads','Multiple heads make different learned reads over the same context. The pictured patterns are examples, not assigned head roles.',290)
    for r,idx in enumerate([0,3,1]):
        y=35+r*79;d.circle(75,y+22,21,'Q','gold');d.text(171,y+8,'Head '+str(r+1),20)
        d.records(250,y,5,63,46,29,selected=[idx])
        d.arrow(100,y+22,236,y+22,'green')
        d.arrow(715,y+22,806,144,'blue')
    d.box(813,117,65,54,'Out','green');d.save()

    d=Drawing('01-prefill','Dense causal prefill has many query-key pairs. Cached decoding introduces one query that reads historical records and appends the new key/value.',295)
    d.text(225,32,'Prefill: many queries',24,bold=True);d.text(675,32,'Decode: one new query',24,bold=True);d.divider(450)
    for r in range(6):
        for c in range(6):
            d.box(119+c*31,65+r*31,25,25,color='blue' if c<=r else 'line',fill=c<=r,opacity=.8 if c<=r else 1)
    d.text(225,285,'Pairwise work ~ N²',21,'blue')
    d.circle(670,90,25,'Q','gold');d.records(489,178,5,50,51,18)
    for i in range(5):d.arrow(670,117,514+i*68,174,'blue')
    d.arrow(687,111,846,174,'green');d.box(823,178,47,51,'+','green')
    d.text(674,285,'Read history; append KV',21,'blue');d.save()


def sharing():
    d=Drawing('02-sharing','Eight query heads use eight KV heads in MHA, two in illustrated GQA, or one in MQA. Each query still forms its own read.',320)
    for panel,(name,nkv) in enumerate([('MHA',8),('GQA',2),('MQA',1)]):
        x=panel*300;d.text(x+150,31,name,25,bold=True)
        if panel:d.divider(x)
        kvys=[75+i*26 for i in range(8)] if nkv==8 else ([111,215] if nkv==2 else [163])
        for i in range(8):
            y=75+i*26;dest=i if nkv==8 else (i//4 if nkv==2 else 0)
            d.arrow(x+74,y,x+188,kvys[dest],'blue');d.circle(x+58,y,9,color='gold',fill=True)
        for y in kvys:d.box(x+194,y-10,46,20,color='blue',fill=True)
        d.text(x+150,309,'8 Q → '+str(nkv)+' KV',23,'green',True)
    d.save()

    d=Drawing('02-cache-size','For the same eight query heads, holding context length, dimensions and precision fixed, KV element counts are proportional to eight, two or one KV heads.',270)
    for i,(label,n) in enumerate([('MHA · 8',8),('GQA · 2',2),('MQA · 1',1)]):
        y=30+i*70;d.text(35,y+29,label,22,anchor='start');d.box(220,y,600*n/8,42,color='blue' if i==0 else 'green',fill=True);d.text(845,y+30,str(n),23,'green',True)
    d.text(450,262,'Same context · same head width · same precision',21,'muted');d.save()

    d=Drawing('02-growth','Sharing reduces width, but the number of retained position records still grows. Twice the retained history doubles this cache component.',260)
    d.text(33,52,'N = 4',23,anchor='start');d.records(180,20,4,60,53,17)
    d.text(33,152,'N = 8',23,anchor='start');d.records(180,120,8,60,53,17)
    d.save()


def latent():
    d=Drawing('03-records','MLA caches one learned latent plus a positional key per historical position rather than expanded content keys and values for every head. Widths are schematic.',300)
    for i in range(4):
        x=102+i*216;d.text(x+33,30,'Position '+str(i+1),21,'muted')
        for r,label in enumerate(['K₁ K₂','V₁ V₂']):d.box(x-43,61+r*44,155,34,label,'blue')
        d.arrow(x+33,145,x+33,190,'purple');d.box(x-13,201,61,50,'c','purple');d.box(x+55,201,35,50,'R','gold')
    d.text(450,288,'One latent c + positional key R at every position',23,'green',True);d.save()

    d=Drawing('03-absorption','Content projection absorption moves key expansion to the query side and value projection after the weighted latent sum. It is algebraically equivalent for the chosen parameterization.',310)
    d.text(35,40,'Keys',23,bold=True,anchor='start');d.text(450,42,'qᵀ(Ac) = (Aᵀq)ᵀc',27,'purple')
    for x,label,c in [(90,'Query q','gold'),(305,'Transform q','purple'),(560,'Match latent c','blue'),(795,'Score','green')]:d.box(x-67,77,134 if x!=560 else 170,49,label,c,size=19)
    d.arrow(160,102,234,102);d.arrow(375,102,489,102);d.arrow(667,102,724,102)
    d.text(35,184,'Values',23,bold=True,anchor='start');d.text(450,186,'Σ w(Uc) = U(Σ wc)',27,'purple')
    for x,w,label,c in [(30,163,'Stored latents','purple'),(273,176,'Weighted sum','blue'),(531,172,'Project result','purple'),(784,89,'Out','green')]:d.box(x,221,w,49,label,c,size=20)
    d.arrow(195,246,269,246);d.arrow(452,246,527,246);d.arrow(706,246,780,246);d.save()

    d=Drawing('03-position','MLA matches content through a latent and position through a separate RoPE key. The two scores combine before softmax.',285)
    d.box(35,47,169,53,'Content query','gold',size=20);d.box(35,166,169,53,'Position query','gold',size=20)
    d.box(321,47,153,53,'Latent c','purple');d.box(321,166,153,53,'RoPE key R','gold')
    d.arrow(207,73,317,73,'purple');d.arrow(207,192,317,192,'gold')
    d.arrow(479,73,651,131,'purple');d.arrow(479,192,651,145,'gold');d.circle(680,137,26,'+','green');d.arrow(709,137,770,137,'green');d.box(777,111,99,52,'Softmax','green',size=19)
    d.text(450,270,'Cache = c + R',26,'green',True);d.save()


def recurrent():
    d=Drawing('04-aggregate','Kernel linear attention accumulates feature-mapped key/value contributions into fixed-size state. A normalized variant also keeps a normalizer.',300)
    d.records(35,32,6,90,45,50)
    for i in range(6):d.arrow(80+i*140,80,395,162,'purple')
    d.grid(351,171,3,4,27,'purple');d.text(405,286,'State S',23,'purple',True)
    d.circle(652,211,28,'Q','gold');d.arrow(620,211,465,211,'gold');d.arrow(461,237,715,267,'green');d.text(798,280,'Read',24,'green',True)
    d.text(170,221,'Accumulate',22,'muted');d.save()

    d=Drawing('04-collision','Different associations can overlap in the same state coordinates. Bounded state has finite capacity; similar keys can interfere.',275)
    d.box(40,38,169,50,'Order 7314','blue');d.box(40,176,169,50,'Order 7315','red')
    d.arrow(212,62,401,111,'blue');d.arrow(212,200,401,138,'red')
    d.grid(410,78,4,4,30,'purple')
    d.box(441,109,27,27,color='red',fill=True,opacity=.75)
    d.arrow(533,137,665,137,'purple');d.box(673,105,185,62,'Mixed read?','red')
    d.save()

    d=Drawing('04-decay','RetNet uses fixed retention factors per head. GLA makes state forgetting input dependent; the illustrated variant decays key channels separately.',310)
    d.text(225,33,'RetNet: fixed time scales',23,bold=True);d.text(675,33,'GLA: input-dependent decay',23,bold=True);d.divider(450)
    for i in range(6):d.box(50+i*58,85,45,57,color='blue',fill=True,opacity=.7**(5-i))
    d.text(72,177,'Old',20,'muted');d.text(363,177,'New',20,'muted')
    d.text(225,254,'One fixed factor per head',21,'blue')
    d.box(530,65,109,43,'Input','gold');d.arrow(643,86,689,86,'gold');d.text(752,93,'Gates',21,'gold')
    d.grid(598,143,3,5,32,'green',[[.15,.9,.4,.85,.25]]*3)
    d.text(675,287,'Different channels can persist.',21,'green');d.save()


def editing():
    d=Drawing('05-residual','Toy delta update for a unit key: old prediction 2, target 5, residual 3. With update strength one, the new prediction is 5.',250)
    for x,label,value,color in [(115,'Old prediction','2','blue'),(355,'Incoming target','5','green'),(595,'Residual','+3','gold'),(820,'New prediction','5','green')]:
        d.text(x,37,label,20,color);d.circle(x,121,42,value,color,True,32)
    d.arrow(402,121,548,121,'gold');d.text(475,92,'5 − 2',22,'gold')
    d.path([(115,165),(115,187),(595,187),(595,167)],'blue',True)
    d.arrow(641,121,772,121,'green');d.text(705,92,'2 + 3',22,'green')
    d.text(450,230,'β = 1',22,'green',True);d.save()

    d=Drawing('05-fade-correct','Toy Gated DeltaNet: scalar decay 0.5 changes state [2,4] to [1,2]. Reading the faded first key gives residual 5-1=4; a full delta correction gives [5,2].',260)
    for x,values,label in [(90,[2,4],'Old state'),(370,[1,2],'Fade by 0.5'),(690,[5,2],'Correct first key')]:
        d.text(x+64,37,label,22,bold=True)
        for j,v in enumerate(values):d.box(x+j*70,91,58,64,str(v),'green' if j==0 and x==690 else 'blue',size=29)
    d.arrow(225,122,360,122,'gold');d.arrow(511,122,679,122,'green');d.text(590,97,'5 − 1 = 4',22,'green')
    d.save()

    d=Drawing('05-channel-decay','A scalar decay fades all state columns; KDA can give key channels different decay rates. The pictured toy directions are separate; learned directions can overlap.',300)
    d.text(225,33,'GDN: one decay per head',23,bold=True);d.text(675,33,'KDA: decay per key channel',23,bold=True);d.divider(450)
    old=[[.25+.65*((r*7+c*3)%9)/8 for c in range(4)] for r in range(4)]
    d.grid(73,84,4,4,34,'blue',old);d.arrow(219,150,276,150,'gold');d.grid(286,84,4,4,34,'blue',[[v*.35 for v in row] for row in old])
    d.grid(523,84,4,4,34,'blue',old);d.arrow(669,150,726,150,'gold');d.grid(736,84,4,4,34,'green',[[v*[.2,1,.4,.9][c] for c,v in enumerate(row)] for row in old])
    d.text(225,272,'All columns fade together.',21,'muted');d.text(675,272,'Some fade; some persist.',21,'green');d.save()


def sparse():
    d=Drawing('06-patterns','Illustrative causal visibility graphs compare dense connections, local plus global links, and the added random links used by BigBird. Original document models also use bidirectional forms.',295)
    n=7
    for panel,name in enumerate(['Dense','Local + global','+ random links']):
        x=40+panel*300;d.text(x+112,33,name,23,bold=True)
        for r in range(n):
            for c in range(n):
                visible=c<=r
                active=visible and (panel==0 or c==0 or r-c<=1 or (panel==2 and (r,c) in [(4,1),(5,2),(6,3)]))
                d.box(x+c*31,65+r*27,25,21,color='green' if active else 'line',fill=active)
    d.save()

    d=Drawing('06-blocks','MoBA scores mean-key summaries, selects historical blocks, and reads original entries inside them. The current block is included causally; unselected history remains stored.',300)
    d.circle(450,36,24,'Q','gold')
    for i,(label,selected) in enumerate([('Past 1',True),('Past 2',False),('Past 3',True),('Current',True)]):
        x=30+i*225;c='green' if selected else 'line'
        if i<3:
            d.circle(x+86,115,18,color=c,fill=selected);d.arrow(450,63,x+86,91,'green' if selected else 'muted',not selected);d.arrow(x+86,136,x+86,177,c if selected else 'muted')
        else:d.arrow(462,62,x+86,177,'gold')
        for j in range(4):d.box(x+j*44,186,33,46,color=c,fill=selected)
        d.text(x+86,270,label,21,c)
    d.save()

    d=Drawing('06-minference','MInference approximates attention during inference using sparse head patterns, primarily accelerating prefill without retraining the model.',275)
    for panel,name in enumerate(['Existing dense attention','Sparse prefill approximation']):
        x=95+panel*460;d.text(x+117,33,name,23,bold=True)
        for r in range(7):
            for c in range(7):
                active=c<=r and (panel==0 or c==0 or r-c<=1 or c==2)
                d.box(x+c*32,65+r*24,26,18,color='blue' if active else 'line',fill=active)
    d.arrow(352,145,522,145,'gold');d.save()

    d=Drawing('06-nsa','NSA trains three gated read branches: compressed summaries, selected original blocks and a local window. Detailed selectable history remains stored.',310)
    d.circle(70,144,26,'Q','gold')
    for y,name in [(30,'Compressed'),(123,'Selected detail'),(217,'Recent window')]:
        d.text(485,y-10,name,21);d.path([(99,144),(122,144),(122,y+22),(322,y+22)],'gold',True)
        if y==30:d.records(333,y,4,59,41,28,selected=[0,1,2,3])
        elif y==123:
            for j in range(8):d.box(333+j*36,y,27,41,color='green' if j in [0,1,4,5] else 'line',fill=j in [0,1,4,5])
        else:d.records(481,y,2,59,41,28,selected=[0,1])
        d.arrow(664,y+22,764,144,'green')
    d.circle(796,144,29,'Σ','green');d.text(797,199,'Gated mix',21,'green');d.save()

    d=Drawing('06-indexer','DSA uses a lightweight indexer to select entries for the main attention. All pictured historical records remain retained; selector work grows with history.',295)
    scores=[.12,.48,.2,.9,.14,.32,.76,.18,.25,.1]
    d.text(150,33,'Cheap indexer',23,'gold',True);d.text(676,33,'Main attention',23,'green',True)
    for i,s in enumerate(scores):
        x=37+i*42;d.box(x,173-s*105,27,s*105,color='green' if i in [3,6] else 'line',fill=True)
        d.box(x,190,27,39,color='green' if i in [3,6] else 'line')
    d.arrow(480,150,582,150,'green');d.box(609,110,64,65,'#4','green');d.box(709,110,64,65,'#7','green')
    d.text(234,272,'Retained history',22,'muted');d.text(691,272,'Selected read',22,'green');d.save()


def pooling():
    d=Drawing('07-pool','MLA retains a compact record at each position. A simple pooling example maps twelve positions to three group records. Pooling diagrams are schematic, not complete V4 cache layouts.',285)
    d.text(225,32,'MLA: shrink each record',23,bold=True);d.text(675,32,'Pool: keep fewer records',23,bold=True);d.divider(450)
    for i in range(12):
        x=35+i*32;d.box(x,70,24,51,color='blue');d.arrow(x+12,125,x+12,171,'purple');d.box(x+4,181,16,37,color='purple',fill=True)
    for i in range(12):d.box(492+i*32,70,23,51,color='blue')
    for i in range(3):
        for j in range(4):d.arrow(504+(i*4+j)*32,125,550+i*128,174,'green')
        d.box(525+i*128,183,50,38,'c','green')
    d.text(225,272,'One per position',22,'purple');d.text(675,272,'One per group',22,'green');d.save()

    d=Drawing('07-csa-hca','CSA selects moderately compressed history. HCA reads every eligible heavily compressed historical entry. Both also read recent uncompressed positions. Counts are illustrative.',305)
    for panel,name in enumerate(['CSA: select compressed entries','HCA: read all compressed entries']):
        y=28+panel*145;d.text(30,y,name,23,bold=True,anchor='start');d.circle(75,y+66,22,'Q','gold')
        n=6 if panel==0 else 3
        for i in range(n):
            x=200+i*60;active=panel==1 or i in [1,4]
            d.box(x,y+44,43,43,'c','green' if active else 'line')
            if active:d.path([(100,y+66),(140,y+66),(140,y+30),(x+21,y+30),(x+21,y+40)],'green',True)
        d.line(645,y+35,645,y+102);d.records(689,y+44,3,45,43,14,selected=[0,1,2])
        d.path([(96,y+80),(125,y+100),(675,y+100),(675,y+65),(685,y+65)],'gold',True);d.text(770,y+124,'Recent detail',20,'gold')
    d.save()

    d=Drawing('07-overlap','CSA ratio four emits every four positions. Interior source windows overlap eight positions, but output count remains about N divided by four, not N divided by eight.',320)
    d.text(35,31,'Source positions',21,'muted',anchor='start')
    for i in range(12):d.box(148+i*43,48,34,35,str(i+1),'blue',size=18)
    for r,(start,end,label) in enumerate([(0,4,'c₁'),(0,8,'c₂'),(4,12,'c₃')]):
        y=111+r*62;d.text(90,y+24,label,23,'green')
        for i in range(start,end):d.box(148+i*43,y,34,34,color='green',fill=True,opacity=.7)
        d.arrow(707,y+17,766,y+17,'green');d.box(775,y-2,59,38,label,'green')
    d.text(450,306,'Stride 4 → 3 entries from 12 positions.',23,'green',True);d.save()


def hybrids():
    d=Drawing('08-stacks','One repeating group in Qwen3-Next has three Gated DeltaNet layers plus gated global GQA; Kimi Linear has three KDA layers plus full MLA. Residual and feed-forward operations are omitted.',335)
    for x,name,cell,record in [(60,'Qwen3-Next','Gated DeltaNet','GQA + output gate'),(510,'Kimi Linear','KDA','MLA')]:
        d.text(x+165,33,name,25,bold=True)
        for i in range(4):
            y=61+i*62;d.box(x,y,330,44,cell if i<3 else record,'green' if i<3 else 'purple',size=22)
            if i<3:d.arrow(x+165,y+45,x+165,y+60)
    d.save()

    d=Drawing('08-gates','An output gate modulates this readout after attention. A state gate alters carried memory and therefore future reads. Both can coexist in a recurrent layer.',285)
    for panel,title in enumerate(['Gate the output','Gate the state']):
        x=panel*450;d.text(x+225,33,title,24,bold=True)
        if panel:d.divider(450)
        if panel==0:
            d.box(x+33,105,133,53,'Attention','blue');d.arrow(x+170,131,x+210,131);d.circle(x+239,131,24,'×','gold');d.arrow(x+239,66,x+239,103,'gold');d.text(x+239,55,'Gate',20,'gold');d.arrow(x+268,131,x+316,131);d.box(x+321,105,93,53,'Out','green')
        else:
            d.grid(x+29,91,3,3,27,'purple');d.arrow(x+115,130,x+159,130);d.circle(x+187,130,24,'×','gold');d.arrow(x+187,66,x+187,103,'gold');d.text(x+187,55,'Gate',20,'gold');d.arrow(x+214,130,x+258,130);d.box(x+264,104,149,53,'Update state','green',size=20)
        d.text(x+225,253,'Changes this readout' if panel==0 else 'Changes future reads',22,'green')
    d.save()

    d=Drawing('08-cost','A hybrid retains a fixed-size recurrent component and a growing explicit-cache component. Fewer explicit layers reduce the growing term, but do not remove it.',270)
    for y,n,label in [(40,4,'Short history'),(145,8,'Long history')]:
        d.text(32,y+30,label,21,anchor='start');d.grid(211,y-3,2,3,22,'purple');d.text(301,y+29,'+',25,'muted');d.records(350,y-5,n,48,46,13)
    d.text(244,250,'State',21,'purple');d.text(590,250,'Historical cache still grows',23,'green',True);d.save()


def systems():
    d=Drawing('09-tiles','Naive attention stores full score and probability intermediates in HBM. FlashAttention maintains tiled scores and running softmax/output on chip; dense pairwise arithmetic stays quadratic.',305)
    d.text(225,32,'Write full intermediates',23,bold=True);d.text(675,32,'Compute in tiles',23,bold=True);d.divider(450)
    d.grid(50,75,6,6,23,'blue');d.grid(253,75,6,6,23,'purple');d.arrow(196,142,244,142)
    d.text(119,254,'Scores',21,'blue');d.text(322,254,'Probabilities',21,'purple')
    d.box(512,65,327,162,color='green');d.text(675,91,'On-chip memory',21,'green')
    d.grid(544,117,3,3,23,'blue');d.arrow(621,149,672,149,'green');d.box(684,122,119,47,'Running O','green',size=20)
    d.text(675,270,'FlashAttention',22,'green',True);d.save()

    d=Drawing('09-execution','FLA implements multiple recurrent rules with efficient chunkwise kernels. Ring Attention distributes dense context across devices while circulating KV blocks; scheduling is distinct from the attention rule.',310)
    d.text(225,32,'FLA: implement recurrent rules',23,bold=True);d.text(675,32,'Ring: distribute context',23,bold=True);d.divider(450)
    for i in range(3):
        x=45+i*135;d.box(x,95,98,61,'Chunk '+str(i+1),'blue',size=19)
        if i<2:d.arrow(x+99,125,x+133,125,'purple')
    d.text(225,210,'Pass a state between chunks.',21,'purple')
    for x,y,i in [(562,74,1),(738,74,2),(738,200,3),(562,200,4)]:d.box(x,y,90,49,'GPU '+str(i),'blue',size=20)
    d.arrow(654,99,734,99,'green');d.arrow(782,127,782,196,'green');d.arrow(734,224,655,224,'green');d.arrow(607,197,607,127,'green')
    d.text(695,170,'KV',22,'green',True);d.save()

    d=Drawing('09-storage','Sharing cache sets across layers reduces repeated storage. Quantizing sixteen-bit elements to four bits reduces raw element bits fourfold, before metadata and numerical costs.',300)
    d.text(225,32,'Share across layers',23,bold=True);d.text(675,32,'Use fewer bits',23,bold=True);d.divider(450)
    for i in range(4):
        d.text(28,88+i*43,'L'+str(i+1),18,'muted')
        d.circle(75,81+i*43,12,color='gold',fill=True);d.arrow(91,81+i*43,235,144,'blue')
    d.grid(245,105,3,4,24,'blue');d.text(225,271,'Different queries, one cache',21,'green')
    for i in range(16):d.box(510+i*22,82,17,40,color='blue',fill=True)
    d.text(675,65,'16 bits',22,'blue');d.arrow(675,127,675,171,'gold')
    for i in range(4):d.box(635+i*22,182,17,40,color='green',fill=True)
    d.text(675,270,'4 bits + scales / overhead',22,'green');d.save()

    d=Drawing('09-tiers','Computed compatible prefixes can be retained in GPU, host RAM or SSD. Reuse may save recomputation, while loading has transfer costs. Persistence alone is not semantic memory.',260)
    for x,label,w in [(40,'GPU',174),(353,'RAM',194),(686,'SSD',174)]:
        d.box(x,75,w,99,label,'blue' if label=='GPU' else 'purple',size=27)
        for j in range(3):d.line(x+24,144+j*8,x+w-24,144+j*8,'line')
    d.arrow(220,112,346,112,'green');d.arrow(346,141,220,141,'gold');d.arrow(552,112,679,112,'green');d.arrow(679,141,552,141,'gold')
    d.text(450,58,'Persist',20,'green');d.text(450,219,'Load',20,'gold');d.save()


def frontier():
    d=Drawing('10-write-control','FG-squared-GDN uses tied key/value channel scaling; its plus variant separates the factors. The strips depict scaling factors rather than arbitrary rates for every state element.',290)
    for panel,title in enumerate(['FG²-GDN: tied factors','FG²-GDN+: separate factors']):
        x=panel*450;d.text(x+225,32,title,23,bold=True)
        if panel:d.divider(450)
        for r,label in enumerate(['Key','Value']):
            d.text(x+36,111+r*90,label,21,anchor='start')
            vals=[.25,.9,.45,.7] if panel==0 or r==0 else [.85,.3,.75,.4]
            for j,v in enumerate(vals):d.box(x+125+j*66,76+r*90,45,47,color='green',fill=True,opacity=v)
        if panel==0:
            for j in range(4):d.line(x+147+j*66,125,x+147+j*66,162,'purple',dash=True)
        d.text(x+225,274,'One shared scaling vector' if panel==0 else 'Different key / value vectors',21,'green')
    d.save()

    d=Drawing('10-erase-write','Gated DeltaNet-2 gates the key before the erase read, and gates the incoming value before forming the write target. Their residual corrects the faded state. The gate is not applied after the erase read.',340)
    d.grid(40,79,4,4,28,'purple');d.text(96,230,'Faded state S',21,'purple')
    d.arrow(159,135,345,135,'purple');d.box(353,109,151,53,'Erase read','purple')
    d.box(190,17,108,43,'Key k','blue');d.arrow(301,39,324,39,'gold');d.circle(352,39,23,'b','gold')
    d.path([(379,39),(427,39),(427,104)],'gold',True)
    d.box(188,242,123,46,'Value v','blue');d.arrow(315,265,329,265,'gold');d.circle(356,265,23,'w','gold')
    d.arrow(382,265,429,265,'green');d.box(435,242,151,46,'Write target','green',size=20)
    d.arrow(509,136,626,164,'purple');d.text(554,121,'−',27,'purple')
    d.arrow(591,265,643,200,'green');d.text(630,255,'+',25,'green')
    d.circle(657,174,27,'Δ','green');d.arrow(688,174,752,174,'green')
    d.text(718,150,'× kᵀ',18,'green')
    d.grid(762,120,4,4,26,'green');d.text(810,270,'Update S',22,'green')
    d.text(450,328,'Gate the key before reading; gate the value before writing.',22,'green',True);d.save()

    d=Drawing('10-uncertainty','A one-dimensional Kalman illustration: greater prior uncertainty and a reliable new observation can move an estimate farther toward that observation. Internal memory uncertainty is not answer confidence.',305)
    for panel,(label,point,width,newpoint) in enumerate([('Uncertain memory · reliable update',2,1.3,4.4),('Certain memory · noisy update',2,.25,2.18)]):
        y=61+panel*143;d.text(450,y-27,label,23,bold=True)
        d.line(210,y+35,784,y+35,'line',3)
        oldx=210+(point-1)*130;newx=210+(newpoint-1)*130
        d.line(oldx-width*42,y+8,oldx+width*42,y+8,'purple',7);d.circle(oldx,y+8,8,color='purple',fill=True)
        noise_width=12 if panel==0 else 70
        d.line(730-noise_width,y+8,730+noise_width,y+8,'green',5);d.circle(730,y+8,8,color='green',fill=True);d.text(844,y+15,'New',20,'green')
        d.arrow(oldx,y+35,newx,y+35,'gold');d.circle(newx,y+35,7,color='gold',fill=True)
        d.text(75,y+15,'Prior',21,'purple');d.text(450,y+77,'Large correction' if panel==0 else 'Small correction',21,'gold')
    d.save()

    d=Drawing('10-tests','Five different evaluation questions: correction, interference, retention, evidence access and full-system cost. One successful test does not establish the others.',220)
    names=['Correct?','Interfere?','Retain?','Find source?','At what cost?']
    for i,name in enumerate(names):
        x=20+i*179;d.text(x+80,37,name,21,bold=True)
        if i==0:
            d.box(x+10,79,52,43,'Tue','red',size=18);d.arrow(x+65,100,x+104,100,'green');d.box(x+109,79,52,43,'Fri','green',size=18)
        elif i==1:
            d.box(x+7,70,148,37,'7314','blue');d.box(x+7,126,148,37,'7315','purple')
        elif i==2:
            for j in range(5):d.box(x+6+j*30,91,20,39,color='green' if j in [0,4] else 'line',fill=j in [0,4])
        elif i==3:
            d.box(x+34,65,89,101,color='blue')
            for j in range(4):d.line(x+47,85+j*18,x+109,85+j*18,'green' if j==2 else 'line',3)
        else:
            for j in range(3):d.box(x+18+j*42,137-j*30,26,30+j*30,color='green',fill=True)
    d.save()


def main():
    for draw in (overview, foundations, sharing, latent, recurrent, editing, sparse, pooling, hybrids, systems, frontier):draw()
    (OUT/'inventory.json').write_text(json.dumps(INVENTORY,indent=2)+'\n')
    print(f'Drew {len(INVENTORY)} visual explanations')


if __name__ == '__main__':
    main()
