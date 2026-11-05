"""Report step 0 independently of night-3 prerequisites and E5 s2500."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/polish-splade-2026-10'
report=EXP/'results/summary_5070_init.md'
marker='<!-- density-match-step0 -->'

def read(path):
    try: return json.loads(path.read_text())
    except (OSError,ValueError): return {}

points=[]
for label,step in [('modernbert-init-n2-e5-s01500',1500),('modernbert-init-n2-e5-s02000',2000),('mlm-probe-a-s01875',1875)]:
    result=read(EXP/'results-regimes/quantized'/label/'tiny10.json')
    sparse=read(EXP/'results/sparsity'/f'{label}.json').get(label,{})
    dims=sparse.get('ALL',{}).get('mean')
    if dims is None:
        means=[v['mean'] for k,v in sparse.items() if k!='ALL' and v.get('mean') is not None]
        dims=sum(means)/len(means) if means else None
    score=result.get('mean_ndcg@10') if result.get('n_tasks')==10 else None
    points.append({'label':label,'step':step,'dims':dims,'tiny10':score,'n_tasks':result.get('n_tasks',0)})
lines=[marker,'','## Porównanie przy dopasowanej gęstości — krok 0','','| Model | krok | wymiary/dokument | tiny10 nDCG@10 |','|---|---:|---:|---:|']
for p in points:
    fmt=lambda v:'pending' if v is None else f'{v:.2f}'
    lines.append(f"| {p['label']} | {p['step']} | {fmt(p['dims'])} | {fmt(p['tiny10'])} |")
m,a=points[1],points[2]
if m['tiny10'] is not None and a['tiny10'] is not None:
    delta=m['tiny10']-a['tiny10']
    lines+=['',f"Dopasowane przybliżeniem gęstości: M s2000 ({m['dims']:.1f}) vs A s1875 ({a['dims']:.1f}); **M − A = {delta:+.2f} pkt nDCG@10**.",
            'To porównanie historycznych sond o zbliżonej gęstości, a nie nowa bramka GM parowanego biegu na świeżych danych. '
            'A s1875 (~391) mieści się w paśmie 327–458, podobnie jak M s2000 (~404).']
lines+=['','E5 s2500 jest wyłączony z werdyktów porównawczych. Wznowienie zmieniło obserwowaną gęstość; '
        'log zgłasza jednak przywrócenie regulatora z kroku 2000 (lambda_d=0.001536), '
        'więc utrata stanu pozostaje hipotezą, nie potwierdzoną przyczyną.',
        'Decyzja: koniec sweepów LR/celu na starych danych; następne porównanie ma użyć świeżych danych i obu backbone’ów. '
        'Podobieństwo profilu błędów wskazuje kierunek badań nad danymi, ale samo nie dowodzi sufitu architektury.','']
text=report.read_text()
text=text.split(marker)[0].rstrip()+'\n\n'+'\n'.join(lines)
temporary=report.with_suffix('.density-match.tmp')
temporary.write_text(text);temporary.replace(report)
output=EXP/'results/data/density-match-step0.json'
output.write_text(json.dumps({'points':points,'excluded':'E5 s2500'},indent=2)+'\n')
print('\n'.join(lines))
