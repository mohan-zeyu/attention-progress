#!/usr/bin/env python3
"""Build the overview and systems diagrams using only standard-library SVG."""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parents[1]
INK = "#203044"
COLORS = {"blue": "#e5edf8", "green": "#eaf5f0", "purple": "#f1edfa", "orange": "#fbf1e5", "plain": "#f4f7fb"}


class Figure:
    def __init__(self, width, height, title, description):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#7d8da4" stroke-width="1.5"/></marker></defs>',
            f'<rect width="{width}" height="{height}" rx="12" fill="white"/>']

    def text(self, x, y, lines, size=17, bold=False, anchor="middle", color=INK):
        lines = [lines] if isinstance(lines, str) else lines
        for n, line in enumerate(lines):
            self.parts.append(f'<text x="{x}" y="{y+n*(size+9)}" text-anchor="{anchor}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{650 if bold else 400}" fill="{color}">{escape(line)}</text>')

    def box(self, x, y, w, h, color="plain"):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{COLORS[color]}" stroke="#c8d3e2"/>')

    def arrow(self, points, dashed=False):
        path = "M" + "L".join(f"{x} {y}" for x,y in points)
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        self.parts.append(f'<path d="{path}" fill="none" stroke="#7d8da4" stroke-width="1.8" marker-end="url(#arrow)"{dash}/>')

    def save(self, name):
        path = ROOT / "figures" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("\n".join(self.parts+["</svg>\n"]))


def overview():
    f = Figure(1000, 875, "Attention: organizations, choices and examples", "First distinguish growing addressable records from bounded recurrent decode state. MQA and GQA share heads, MLA shrinks each position's record, and CSA and HCA pool positions. Kernel linear attention, RetNet, GLA and delta mechanisms update aggregate state. Hybrid layer stacks combine the organizations. Lower lanes separate representation, update, retrieval and execution choices; they are not a flat list of alternative algorithms.")
    f.text(500, 35, "Preserve and use context within a compute, storage and bandwidth budget", 20, True)
    f.text(25, 90, "1 · CHOOSE HOW HISTORY IS RETAINED", 17, True, "start")
    f.box(25, 112, 460, 195, "blue")
    f.text(255, 146, "ADDRESSABLE RECORDS", 19, True)
    f.text(255, 181, "A growing sequence of separate entries", 17, color="#52647b")
    f.text(255, 221, ["Share KV heads: MQA / GQA", "Shrink each position's record: MLA", "Pool positions: CSA / HCA"], 17)
    f.box(515, 112, 460, 195, "purple")
    f.text(745, 146, "BOUNDED RECURRENT STATE", 19, True)
    f.text(745, 181, "One aggregate state, fixed at decode", 17, color="#52647b")
    f.text(745, 221, ["Accumulate: kernel linear attention", "Decay: RetNet / GLA", "Correct: DeltaNet / GDN / KDA"], 17)
    f.arrow([(255,307),(255,325),(340,325),(340,336)])
    f.arrow([(745,307),(745,325),(660,325),(660,336)])
    f.box(220, 340, 560, 80, "plain")
    f.text(500, 372, "HYBRID MODEL DESIGNS", 19, True)
    f.text(500, 401, "Layer stacks combine the two organizations", 17)
    f.text(25, 470, "2 · DESCRIBE THE CHOICES WITHIN THE ORGANIZATION", 17, True, "start")
    columns = [(25,"blue","REPRESENTATION","What is retained?",["Records: one / many positions","State: shape and capacity"]),
               (265,"purple","UPDATE","How does it change?",["Records: append / pool","State: add / decay / correct","FG² / GDN-2 / KDN*"]),
               (505,"green","RETRIEVAL","How is memory read?",["Records: dense / sparse","MoBA / NSA / DSA","State: read aggregate"]),
               (745,"orange","EXECUTION / STORAGE","How is it realized?",["Tiling / distribution","Reuse / precision / placement","FlashAttention: kernels","FLA: implementation library"])]
    for x,color,title,question,lines in columns:
        f.box(x, 488, 230, 244, color)
        f.text(x+115, 521, title, 15, True)
        f.text(x+115, 556, question, 15, color="#52647b")
        f.text(x+115, 600, lines, 15)
    f.text(500, 773, "* Research extensions: finer-grained writes, independent editing and uncertainty", 15, color="#5c6d83")
    f.box(25, 800, 950, 45, "plain")
    f.text(500, 829, "Supporting choices: relative position / RoPE · output gates", 17)
    f.save("00-memory-map.svg")


def storage():
    f=Figure(1000,365,"Separate cache capacity from reads and arithmetic", "Cache bytes are a product of batch, positions, independent cache sets, dimensions per record and bytes per element. Access count and selector work separately affect traffic and arithmetic.")
    f.text(500,34,"Cache bytes ≈ B × N × C × Dentry × s",22,True)
    data=[("B","Concurrent sequences","Batch / serving load","blue"),("N","Stored records","Sequence compression","green"),("C","Independent caches","Cross-layer reuse","purple"),("Dentry","Elements per record","Sharing / MLA","blue"),("s","Bytes per element","KV quantization","orange")]
    for i,(symbol,label,mechanism,color) in enumerate(data):
        x=20+i*195
        f.box(x,66,180,133,color)
        f.text(x+90,99,symbol,22,True)
        f.text(x+90,134,label,15)
        f.text(x+90,173,mechanism,14,color="#52647b")
    f.text(500,244,"Read traffic: accessed records × record size, plus indices and metadata",17)
    f.text(500,276,"Arithmetic: attention operation + selection + update work",17)
    f.text(500,321,"Recurrent layers: replace growing history with a state of fixed dimensions",16,color="#52647b")
    f.save("09-storage-factors.svg")


def execution():
    f=Figure(1000,520,"Naive versus tiled attention execution", "Naive dense attention writes full score and probability matrices to HBM. Tiled execution loads blocks, maintains online softmax and accumulates output on chip without storing a full N by N intermediate in HBM.")
    f.text(500,34,"Same permitted query–key pairs; different execution and storage",21,True)
    f.text(25,82,"NAIVE: full intermediate matrices in HBM",17,True,"start")
    boxes=[(25,210,["Q / K / V","input matrices"],"blue"),(280,210,["Scores","N × N"],"orange"),(535,210,["Probabilities","N × N"],"orange"),(790,185,["Output","N × dv"],"green")]
    for x,w,lines,color in boxes:
        f.box(x,103,w,84,color)
        f.text(x+w/2,135,lines,18)
    for start,end in [(235,280),(490,535),(745,790)]: f.arrow([(start,145),(end-4,145)])
    f.text(25,252,"TILED: only a block of scores is needed on chip at a time",17,True,"start")
    f.box(25,285,215,92,"blue")
    f.text(132,320,["Q / K / V in HBM","load the next tile"],17)
    f.box(315,274,380,151,"purple")
    f.text(505,307,"FAST ON-CHIP MEMORY",18,True)
    f.text(505,340,["Block scores + online softmax","Rescale running output / normalizer","Accumulate values; then repeat"],16)
    f.box(775,285,200,92,"green")
    f.text(875,320,["Final output","written to HBM"],17)
    f.arrow([(240,330),(311,330)])
    f.arrow([(695,330),(771,330)])
    f.text(500,477,"Dense pairwise arithmetic stays quadratic; large intermediate HBM writes disappear",16,color="#52647b")
    f.save("09-kernel-execution.svg")


if __name__ == "__main__":
    overview()
    storage()
    execution()
    print("Built overview and systems diagrams")
