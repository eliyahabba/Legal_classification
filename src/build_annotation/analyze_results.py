# -*- coding: utf-8 -*-
"""
Analyze annotation results: produce summary tables (CSV) and clean figures (PNG).

Usage:
    python analyze_results.py results_renana_exp9.json results_carmit_exp9.json [classification_results.csv]

If the CSV (the experiment pool, for sentence text) is omitted, it is found from
the experiment field via data/experiments/<exp>/classification_results.csv.
Outputs land in ./analysis_<exp>/ next to the script.
"""
import json, sys, collections
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
from matplotlib import font_manager as fm, pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = Path(__file__).resolve().parent
FONT = SCRIPT_DIR / "fonts" / "Heebo.ttf"

# ---- Hebrew rendering ----
try:
    from bidi.algorithm import get_display
except Exception:
    def get_display(s): return s
if FONT.is_file():
    fm.fontManager.addfont(str(FONT)); plt.rcParams["font.family"] = "Heebo"
plt.rcParams["axes.unicode_minus"] = False
def H(s): return get_display(str(s))
import textwrap
def LAB(s, w=34):
    parts = textwrap.wrap(str(s), width=w) or [""]
    return "\n".join(get_display(p) for p in parts)

# ---- palette ----
PAPER="#F7F5F0"; INK="#1F1D1A"; MUT="#6B6660"; LINE="#E4DFD5"
ACCENT="#6E2B2B"; YES="#2E6B4F"; NO="#A14233"
COL={"renana":"#3E6B89","carmit":"#B07A2C"}      # per-annotator colors
plt.rcParams.update({"figure.facecolor":PAPER,"axes.facecolor":PAPER,"savefig.facecolor":PAPER,
                     "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
                     "axes.edgecolor":LINE})

# ---- fixed category order + short labels ----
ACCEPT=["קבלת עדות המתלוננת במלואה לאור רושם חיובי, כנות ומהימנות גבוהה",
        "קבלת עדות המתלוננת לאור תיאור מפורט, אותנטי וברור של מסכת האירועים",
        "קבלת גרסת המתלוננת בהיותה עדות מהימנה הנתמכת בראיות סיוע וחיזוקים",
        "קבלת עדות הקטינה כעדות אמת מהימנה המגובה בהתרשמות חוקרת הילדים",
        "העדפת עדותה המהימנה של המתלוננת ודחיית גרסת הנאשם עקב חוסר אמינותו",
        "קביעת ממצאים עובדתיים מרשיעים על בסיס אימוץ גרסת המתלוננת ודחיית הנאשם"]
REJECT=["דחיית עדות המתלוננת עקב חוסר עקביות וסתירות מהותיות בגרסתה",
        "דחיית הרשעה על בסיס עדות יחידה של המתלוננת בהעדר ראיות סיוע"]
NEUTRAL=["לא רלוונטי"]
CATS=ACCEPT+REJECT+NEUTRAL
SHORT={ACCEPT[0]:"קבלה — רושם חיובי", ACCEPT[1]:"קבלה — פירוט ואותנטיות",
       ACCEPT[2]:"קבלה — ראיות סיוע", ACCEPT[3]:"קבלה — עדות קטינה",
       ACCEPT[4]:"העדפה על הנאשם", ACCEPT[5]:"ממצאים מרשיעים",
       REJECT[0]:"דחייה — חוסר עקביות", REJECT[1]:"דחייה — עדות יחידה",
       NEUTRAL[0]:"לא רלוונטי", "אחר (פרטי בהערה)":"אחר"}
def sh(c): return SHORT.get(c, str(c)[:18])

# ---- load ----
args=[a for a in sys.argv[1:]]
jsons=[a for a in args if a.endswith(".json")]
csvs =[a for a in args if a.endswith(".csv")]
if len(jsons)<2: sys.exit("pass two results JSON files")
res=[json.loads(Path(p).read_text(encoding="utf-8")) for p in jsons]
# Normalize: if marked "incorrect" but the chosen category equals the model's,
# that is actually agreement with the model -> treat as "correct".
for _r in res:
    for _x in _r["answers"]:
        if _x.get("decision")=="incorrect" and _x.get("corrected")==_x["assigned"]:
            _x["decision"]="correct"; _x["corrected"]=None
exp=res[0].get("experiment","")
def find_root():
    for s in (SCRIPT_DIR, Path.cwd().resolve()):
        for d in (s,*s.parents):
            if (d/"data"/"experiments").is_dir(): return d
    return None
csv = Path(csvs[0]) if csvs else (find_root()/"data"/"experiments"/exp/"classification_results.csv" if (find_root() and exp) else None)
pool={}
if csv and Path(csv).is_file():
    df=pd.read_csv(csv).drop_duplicates("sentence_id"); df["sentence_id"]=df["sentence_id"].astype(str)
    pool=df.set_index("sentence_id").to_dict("index")
def text_of(sid):
    p=pool.get(str(sid),{}); v=p.get("origin_sentence","")
    return "" if (v is None or (isinstance(v,float) and pd.isna(v))) else str(v).strip()

OUT=SCRIPT_DIR/f"analysis{('_'+exp) if exp else ''}"; OUT.mkdir(exist_ok=True)
NAMES=[r.get("annotator_name",r["annotator_id"]) for r in res]
IDS=[r["annotator_id"] for r in res]
def final(x): return x["corrected"] if x["decision"]=="incorrect" else x["assigned"]

# ---- stats ----
def per_annotator(r):
    a=[x for x in r["answers"] if x["decision"] is not None]
    c=sum(1 for x in a if x["decision"]=="correct")
    return {"total":r.get("total",len(r["answers"])),"completed":len(a),"correct":c,
            "incorrect":len(a)-c,"acc":c/len(a) if a else 0}
def percat(r):
    m={cat:[0,0] for cat in CATS}
    for x in r["answers"]:
        if x["decision"] is None or x["assigned"] not in m: continue
        m[x["assigned"]][0]+=1
        if x["decision"]=="correct": m[x["assigned"]][1]+=1
    return m
def kappa(pairs):
    N=len(pairs)
    if not N: return (0,0,0)
    ag=sum(1 for a,b in pairs if a==b); Po=ag/N
    ca=collections.Counter(a for a,_ in pairs); cb=collections.Counter(b for _,b in pairs)
    Pe=sum((ca[l]/N)*(cb[l]/N) for l in set(list(ca)+list(cb)))
    return (N,Po,(Po-Pe)/(1-Pe) if Pe<1 else 1.0)

S=[per_annotator(r) for r in res]
PC=[percat(r) for r in res]

# shared agreement (first two annotators)
A,B=res[0],res[1]
da={x["sentence_id"]:x for x in A["answers"] if x["decision"] is not None}
db={x["sentence_id"]:x for x in B["answers"] if x["decision"] is not None}
both=[s for s in da if s in db]
decp=[(da[s]["decision"],db[s]["decision"]) for s in both]
labp=[(final(da[s]),final(db[s])) for s in both]
kd=kappa(decp); kl=kappa(labp)

# ---------- TABLES (CSV) ----------
pd.DataFrame([{"annotator":NAMES[i],"total":S[i]["total"],"completed":S[i]["completed"],
               "correct":S[i]["correct"],"incorrect":S[i]["incorrect"],
               "accuracy_%":round(100*S[i]["acc"],1)} for i in range(len(res))]
            ).to_csv(OUT/"summary_per_annotator.csv",index=False)

rows=[]
for cat in CATS:
    row={"category":cat}
    for i in range(len(res)):
        t,c=PC[i][cat]; row[f"{IDS[i]}_correct"]=c; row[f"{IDS[i]}_total"]=t
        row[f"{IDS[i]}_acc_%"]=round(100*c/t,1) if t else ""
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT/"accuracy_by_category.csv",index=False)

pd.DataFrame([{"metric":"shared_items_n","value":kd[0]},
              {"metric":"decision_raw_agreement_%","value":round(100*kd[1],1)},
              {"metric":"decision_cohen_kappa","value":round(kd[2],3)},
              {"metric":"final_label_raw_agreement_%","value":round(100*kl[1],1)},
              {"metric":"final_label_cohen_kappa","value":round(kl[2],3)}]
            ).to_csv(OUT/"agreement.csv",index=False)

note_rows=[]
for r in res:
    for x in r["answers"]:
        if x.get("note","").strip():
            note_rows.append({"category":x["assigned"],"annotator":r.get("annotator_name"),
                              "sentence_id":x["sentence_id"],"decision":x["decision"],
                              "corrected":x.get("corrected") or "","note":x["note"].strip(),
                              "sentence":text_of(x["sentence_id"])})
notes_df=pd.DataFrame(note_rows).sort_values(["category","annotator"]) if note_rows else pd.DataFrame()
notes_df.to_csv(OUT/"notes.csv",index=False)

# ---------- FIGURES ----------
def style(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(LINE); ax.spines["bottom"].set_color(LINE)
    ax.tick_params(length=0)

# Fig 1: overall agreement with the model
fig,ax=plt.subplots(figsize=(7.5,4.2)); style(ax)
ax.set_axisbelow(True); ax.grid(axis="y",color=LINE,lw=.9)
xs=range(len(res)); accs=[S[i]["acc"]*100 for i in range(len(res))]
bars=ax.bar(xs,accs,width=.55,color=[COL.get(IDS[i],ACCENT) for i in range(len(res))])
for i,b in enumerate(bars):
    ax.text(b.get_x()+b.get_width()/2,accs[i]+1.5,H(f"{accs[i]:.0f}%"),ha="center",fontsize=15,fontweight="bold")
    ax.text(b.get_x()+b.get_width()/2,accs[i]/2,H(f"{S[i]['correct']}/{S[i]['completed']}"),ha="center",va="center",color="white",fontsize=12)
ax.set_xticks(list(xs)); ax.set_xticklabels([H(n) for n in NAMES],fontsize=13)
ax.set_ylim(0,100); ax.set_ylabel(H("אחוז הסכמה עם המודל")); ax.set_yticks([0,25,50,75,100])
ax.set_yticklabels([H(f"{v}%") for v in [0,25,50,75,100]])
ax.set_title(H("הסכמה עם תיוג המודל"),fontsize=16,fontweight="bold",pad=14)
fig.tight_layout(); fig.savefig(OUT/"fig1_model_agreement.png",dpi=150); plt.close(fig)

# Fig 2: per-category accuracy, grouped horizontal bars (sorted by mean acc)
order=sorted(CATS,key=lambda c:sum((PC[i][c][1]/PC[i][c][0] if PC[i][c][0] else 0) for i in range(len(res))))
y=range(len(order)); hh=0.38
fig,ax=plt.subplots(figsize=(13,7)); style(ax)
ax.set_axisbelow(True); ax.grid(axis="x",color=LINE,lw=.9)
for i in range(len(res)):
    vals=[(PC[i][c][1]/PC[i][c][0]*100 if PC[i][c][0] else 0) for c in order]
    off=(i-0.5)*hh
    b=ax.barh([yy+off for yy in y],vals,height=hh,color=COL.get(IDS[i],ACCENT),label=H(NAMES[i]))
    for yy,v,c in zip(y,vals,order):
        t,cc=PC[i][c]; ax.text(v+1,yy+off,H(f"{v:.0f}% ({cc}/{t})"),va="center",fontsize=9,color=MUT)
ax.set_yticks(list(y)); ax.set_yticklabels([LAB(c,42) for c in order],fontsize=9)
ax.set_xlim(0,118); ax.set_xticks([0,25,50,75,100]); ax.set_xticklabels([H(f"{v}%") for v in [0,25,50,75,100]])
ax.set_title(H("דיוק המודל לפי קטגוריה (לפי המתייגות)"),fontsize=16,fontweight="bold",pad=14)
ax.legend(loc="lower right",frameon=False,fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"fig2_accuracy_by_category.png",dpi=150); plt.close(fig)

# Fig 3: inter-annotator agreement — clean summary table
from matplotlib.patches import Rectangle
fig,ax=plt.subplots(figsize=(11.5,3.7)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
fig.suptitle(H("הסכמה בין המתייגים"),fontsize=17,fontweight="bold",y=1.02)
edges=[0.0,0.18,0.36,0.50,1.0]     # left->right: kappa | agreement | n | level
def cx(i): return (edges[i]+edges[i+1])/2
headY=(0.74,0.93); r1=(0.50,0.74); r2=(0.26,0.50)
ax.add_patch(Rectangle((0,headY[0]),1,headY[1]-headY[0],color=ACCENT,zorder=1))
for i,htxt in enumerate(["Cohen's kappa", H("הסכמה גולמית"), H("מספר משפטים"), H("רמת ההסכמה")]):
    ax.text(cx(i),sum(headY)/2,htxt,ha="center",va="center",color="white",fontsize=12.5,fontweight="bold",zorder=2)
def row(yr,bg,level,agree,kap,n):
    ax.add_patch(Rectangle((0,yr[0]),1,yr[1]-yr[0],color=bg,zorder=1))
    kc=YES if kap>=0.6 else (ACCENT if kap>=0.2 else NO)
    ax.text(cx(0),sum(yr)/2,f"{kap:.2f}",ha="center",va="center",fontsize=21,fontweight="bold",color=kc,zorder=2)
    ax.text(cx(1),sum(yr)/2,H(f"{agree*100:.0f}%"),ha="center",va="center",fontsize=21,fontweight="bold",color=INK,zorder=2)
    ax.text(cx(2),sum(yr)/2,H(str(n)),ha="center",va="center",fontsize=17,color=MUT,zorder=2)
    ax.text(edges[4]-0.02,sum(yr)/2,LAB(level,52),ha="right",va="center",ma="right",fontsize=11.5,color=INK,zorder=2)
row(r1,"#FFFFFF","הסכמה על ההחלטה — האם תיוג המודל נכון (נכון / שגוי)",kd[1],kd[2],kd[0])
row(r2,"#F1EEE8","הסכמה על הקטגוריה הסופית — התיוג המקורי אם התקבל, או החדש אם נדחה",kl[1],kl[2],kl[0])
ax.add_patch(Rectangle((0,r2[0]),1,headY[1]-r2[0],fill=False,ec=LINE,lw=1.3,zorder=3))
for e in edges[1:-1]: ax.plot([e,e],[r2[0],headY[1]],color=LINE,lw=1,zorder=3)
ax.plot([0,1],[r1[0],r1[0]],color=LINE,lw=1,zorder=3)
fig.savefig(OUT/"fig3_agreement.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# Fig 4: confusion matrix of final labels on shared items
present=[c for c in CATS if any(final(da[s])==c or final(db[s])==c for s in both)]
idx={c:i for i,c in enumerate(present)}; M=[[0]*len(present) for _ in present]
for s in both:
    fa,fb=final(da[s]),final(db[s])
    if fa in idx and fb in idx: M[idx[fa]][idx[fb]]+=1
fig,ax=plt.subplots(figsize=(15,13))
cmap=LinearSegmentedColormap.from_list("ac",["#FFFFFF",COL["renana"]])
im=ax.imshow(M,cmap=cmap)
ax.set_xticks(range(len(present))); ax.set_yticks(range(len(present)))
ax.set_xticklabels([LAB(c,18) for c in present],rotation=45,ha="left",fontsize=8)
ax.set_yticklabels([LAB(c,18) for c in present],fontsize=8)
ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
for i in range(len(present)):
    for j in range(len(present)):
        if M[i][j]: ax.text(j,i,H(str(M[i][j])),ha="center",va="center",
                            color="white" if M[i][j]>max(max(r) for r in M)/2 else INK,fontsize=11)
ax.set_xlabel(H(f"קטגוריה סופית — {NAMES[1]}")); ax.set_ylabel(H(f"קטגוריה סופית — {NAMES[0]}"))
ax.set_title(H("התאמת הקטגוריה הסופית במשפטים המשותפים"),fontsize=14,fontweight="bold",pad=30)
fig.tight_layout(); fig.savefig(OUT/"fig4_shared_confusion.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# Fig 5: where the model erred — both annotators combined.
# label: model's category (red) on top, the category the annotators chose (green) below; bars in blue.
BLUE="#2F6AA0"
err=collections.Counter()
for r in res:
    for x in r["answers"]:
        if x["decision"]=="incorrect":
            err[(x["assigned"], x.get("corrected") or "(לא נבחרה)")]+=1
items=err.most_common()
fig,ax=plt.subplots(figsize=(12,max(4,1.15*len(items)+1.6)))
style(ax); ax.set_axisbelow(True); ax.grid(axis="x",color=LINE,lw=.9)
vals=[n for _,n in items]
ax.barh(range(len(items)),vals,color=BLUE,height=.55)
for i,v in enumerate(vals): ax.text(v+0.06,i,H(str(v)),va="center",fontsize=12,fontweight="bold",color=INK)
ax.set_yticks(range(len(items))); ax.set_yticklabels([""]*len(items))
ax.set_ylim(-0.6,len(items)-0.4); ax.invert_yaxis()
tr=ax.get_yaxis_transform()   # x in axes-fraction, y in data coords
for i,((a,b),_) in enumerate(items):
    ax.text(-0.02,i-0.04,LAB(a,30),transform=tr,ha="right",va="bottom",fontsize=8.5,color=NO)
    ax.text(-0.02,i+0.06,LAB(b,30),transform=tr,ha="right",va="top",fontsize=8.5,color=YES)
ax.set_xlabel(H("מספר מקרים")); ax.set_xticks(range(0,max(vals)+1))
ax.set_title(H("היכן המודל סווג שגוי (לפי אחד המתייגים)"),fontsize=15,fontweight="bold",pad=26)
ax.text(0.5,1.012,H("אדום — התיוג של המודל · ירוק — התיוג הנכון לפי המתייגים"),
        transform=ax.transAxes,ha="center",va="bottom",fontsize=10.5,color=MUT)
fig.subplots_adjust(left=0.46,right=0.97,top=0.9,bottom=0.12)
fig.savefig(OUT/"fig5_model_errors.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---- console summary ----
print(f"experiment: {exp or '(none)'} | output: {OUT}")
for i in range(len(res)):
    print(f"  {NAMES[i]}: {S[i]['correct']}/{S[i]['completed']} correct = {S[i]['acc']*100:.0f}%")
print(f"  shared n={kd[0]} | decision agree={kd[1]*100:.0f}% kappa={kd[2]:.2f} | final-label agree={kl[1]*100:.0f}% kappa={kl[2]:.2f}")
print(f"  notes: {len(note_rows)}")
print("figures + tables written to", OUT.name)
