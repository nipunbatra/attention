"""One diagram source for notebook walkthroughs and the Part II lecture.

The attention graph follows CausalAttentionLM.forward: one supervised final
query per window. forward_details additionally materializes all causal rows.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import json

COLORS = {"data": ("#245EDB", "#E4ECFF"), "q": ("#8B2CDE", "#F1E5FC"),
          "a": ("#BE123C", "#FDE2E7"), "v": ("#0F766E", "#D9F2EF"),
          "update": ("#147737", "#DDF3E4"), "neutral": ("#4A5160", "#EEF0F4")}


def graph(kind="attention", mode="training"):
    if kind not in {"mlp", "attention"} or mode not in {"training", "inference"}:
        raise ValueError("Choose mlp/attention and training/inference")
    attention = kind == "attention"
    bottom = 565 if attention else 400
    height = bottom + 130
    nodes, edges = [], []

    def node(key, x, y, title, detail, role="data", width=230):
        nodes.append(dict(key=key, x=x, y=y, w=width, h=86, title=title,
                          detail=detail, role=role))

    def edge(start, end, points=None, label="", role="neutral", dashed=False, label_pos=None):
        edges.append(dict(start=start, end=end, points=points, label=label,
                          role=role, dashed=dashed, label_pos=label_pos))

    if mode == "training":
        for key, x, title, detail in [
            ("stories",40,"TinyStories","complete documents"),
            ("split",340,"Split stories","train / validation / test"),
            ("tokenize",640,"Tokenize + lookup","train-only vocabulary"),
            ("ids",940,"IDs + boundaries","<BOS> … <EOS>"),
            ("windows",1240,"Context and target","X [B,w]   y [B]")]:
            node(key,x,30,title,detail)
        for a,b in zip(["stories","split","tokenize","ids"],["split","tokenize","ids","windows"]):
            edge(a,b,label="train" if a=="split" else "",label_pos=(605,63) if a=="split" else None)
        edge("windows","embedding",[(1355,116),(1355,165),(155,165),(155,230)])
    else:
        node("prompt",40,30,"Prompt text","known words only")
        node("tokenize",440,30,"Saved tokenizer","same vocabulary + <BOS>")
        node("windows",840,30,"Crop, then left-pad","last w IDs → X [1,w]")
        node("checkpoint",1240,30,"Saved parameters","θ stays fixed","q")
        edge("prompt","tokenize"); edge("tokenize","windows")
        edge("windows","embedding",[(955,116),(955,165),(155,165),(155,230)])
        edge("checkpoint","embedding",[(1355,116),(1355,195),(285,195),(285,210),(200,210),(200,230)],"load weights for every learned layer","q",True,(790,190))

    node("embedding",40,230,"Token lookup" + (" + position" if attention else ""),
         "E [B,w,d]" if mode=="training" else "E [1,w,d]")
    if attention:
        node("qkv",340,230,"Final q, all K and V","q [B,1,dₖ] · K,V: w rows","q")
        node("scores",640,230,"qKᵀ / √dₖ","scores [B,1,w]","a")
        node("weights",940,230,"Mask PAD · softmax","weights A [B,1,w]","a")
        node("message",1240,230,"A @ V","message [B,1,dᵥ]","v")
        node("projection",1240,395,"Output map Wₒ","update [B,1,d]","update")
        node("residual",940,395,"Add original final row","e′ = e + update","update")
        node("readout",640,395,"Final updated vector","[B,d]","update")
        node("hidden",340,395,"Affine + ReLU","hidden [B,h]","v")
        node("logits",40,395,"Vocabulary affine","logits z [B,C]","neutral")
        for a,b in zip(["embedding","qkv","scores","weights"],["qkv","scores","weights","message"]): edge(a,b)
        edge("message","projection",[(1355,316),(1355,395)])
        for a,b in zip(["projection","residual","readout","hidden"],["residual","readout","hidden","logits"]): edge(a,b)
        edge("qkv","message",[(455,316),(455,342),(1355,342),(1355,316)],"V bypasses scoring","v",True)
        edge("embedding","residual",[(155,316),(155,370),(1055,370),(1055,395)],"keep the final input row","data",True)
    else:
        node("flatten",440,230,"Join ordered rows","flatten [B,w·d]")
        node("hidden",840,230,"Affine + ReLU","hidden [B,h]","v")
        node("logits",1240,230,"Vocabulary affine","logits z [B,C]","neutral")
        edge("embedding","flatten");edge("flatten","hidden");edge("hidden","logits")

    if mode == "training":
        node("loss",40,bottom,"Cross-entropy(z, y)","one target per window","a")
        node("backward",440,bottom,"loss.backward()","gradients for θ","q")
        node("optimizer",840,bottom,"optimizer.step()","update learned θ","q")
        node("parameters",1240,bottom,"Next training batch","reuse updated θ","q")
        if attention: edge("logits","loss",[(155,481),(155,bottom)])
        else: edge("logits","loss",[(1355,316),(1355,350),(155,350),(155,bottom)])
        edge("windows","loss",[(1470,73),(1490,73),(1490,bottom-32),(280,bottom-32),(280,bottom+43),(270,bottom+43)],"observed target y [B]","a",True,(1110,bottom-40))
        for a,b in zip(["loss","backward","optimizer"],["backward","optimizer","parameters"]): edge(a,b)
        edge("parameters","embedding",[(1355,bottom+86),(1355,bottom+112),(20,bottom+112),(20,205),(155,205),(155,230)],role="q",dashed=True)
    else:
        node("probabilities",40,bottom,"Vocabulary softmax","z / temperature","a")
        node("choose",440,bottom,"Choose a token","sample or greedy","update")
        node("stop",840,bottom,"Is it <EOS>?","yes: finish · no: append","neutral")
        node("append",1240,bottom,"Append to history","repeat if budget remains","data")
        if attention: edge("logits","probabilities",[(155,481),(155,bottom)])
        else: edge("logits","probabilities",[(1355,316),(1355,350),(155,350),(155,bottom)])
        for a,b in zip(["probabilities","choose","stop"],["choose","stop","append"]): edge(a,b)
        edge("append","windows",[(1470,bottom+43),(1490,bottom+43),(1490,145),(955,145),(955,116)],role="data",dashed=True)
    return dict(kind=kind,mode=mode,width=1520,height=height,nodes=nodes,edges=edges)


def pipeline_svg(kind="attention", mode="training", focus=(), *, region=None):
    """Return a complete, labelled SVG; highlight nodes without moving them."""
    g=graph(kind,mode); active=set([focus] if isinstance(focus,str) else focus)
    keys={n["key"] for n in g["nodes"]}
    if active-keys: raise ValueError("Unknown diagram stage: "+str(active-keys))
    view=region or (0,0,g["width"],g["height"])
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{" ".join(map(str,view))}" role="img" aria-label="{kind} {mode} pipeline" style="font-family:Avenir Next,Segoe UI,sans-serif;background:#F7F8FA">',
         '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="context-stroke"/></marker></defs>']
    by_id={n["key"]:n for n in g["nodes"]}
    for e in g["edges"]:
        a,b=by_id[e["start"]],by_id[e["end"]]
        points=e["points"]
        if points is None:
            right=b["x"]>a["x"]
            points=[(a["x"]+(a["w"] if right else 0),a["y"]+43),(b["x"]+(0 if right else b["w"]),b["y"]+43)]
        stroke=COLORS[e["role"]][0];opacity=1 if not active or e["start"] in active or e["end"] in active else .22
        d="M"+" L".join(f"{x} {y}" for x,y in points)
        out.append(f'<g opacity="{opacity}" data-edge="{e["start"]}:{e["end"]}"><path d="{d}" fill="none" stroke="{stroke}" stroke-width="2.5" marker-end="url(#arrow)"'+(' stroke-dasharray="7 5"' if e["dashed"] else '')+'/>')
        if e["label"]:
            x,y=e.get("label_pos") or ((points[1][0]+points[2][0])/2,points[1][1]-7)
            out.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="21" fill="{stroke}">{escape(e["label"])}</text>')
        out.append('</g>')
    for n in g["nodes"]:
        stroke,fill=COLORS[n["role"]];on=not active or n["key"] in active
        out.append(f'<g data-stage="{n["key"]}" opacity="{1 if on else .25}"><rect x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="86" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="{4 if active and on else 2}"/>')
        # Long labels get two lines rather than smaller text.
        title=n["title"]
        if len(title)>19:
            words=title.split();mid=len(words)//2;lines=[" ".join(words[:mid])," ".join(words[mid:])]
        else: lines=[title]
        for i,line in enumerate(lines):
            out.append(f'<text x="{n["x"]+115}" y="{n["y"]+29+i*25}" text-anchor="middle" font-size="24" font-weight="600" fill="#14171F">{escape(line)}</text>')
        out.append(f'<text x="{n["x"]+115}" y="{n["y"]+75}" text-anchor="middle" font-size="18" fill="#4A5160">{escape(n["detail"])}</text></g>')
    out.append('</svg>');return ''.join(out)


def show_pipeline(kind="attention", mode="training", focus=(), region=None):
    from IPython.display import SVG, display
    display(SVG(pipeline_svg(kind,mode,focus,region=region)))


def export_maps(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for kind in ["mlp","attention"]:
        for mode in ["training","inference"]:
            (destination/f"{kind}-{mode}.svg").write_text(pipeline_svg(kind,mode),encoding="utf8")
    (destination/"maps.json").write_text(json.dumps([graph(k,m) for k in ["mlp","attention"] for m in ["training","inference"]],indent=2),encoding="utf8")


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("destination",type=Path)
    export_maps(parser.parse_args().destination)
