"""Replay released-style BN updates on frozen collapsed checkpoints.

Runs the separate source and unshuffled target train-mode passes for 5 epochs
with no optimizer step, then re-evaluates. Recovery means the saved running
stats were stale (weight drift), not contaminated by target data.

Run: python -m src.experiments.replay_bn_updates
"""
import csv, torch, numpy as np
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset
from src.models.scvae_reference import ReferenceSCVAE
from src.experiments.train_scvae_reference import load_fold, DEFAULT_DATA_DIR
from pathlib import Path
rows=list(csv.DictReader(open('results/diagnostics/collapse/reference_4s/folds.csv')))
def ev(m,x,y):
    m.eval()
    with torch.no_grad(): p=m(x)[3].argmax(1)
    return (p==y).float().mean().item(), torch.bincount(p,minlength=3).max().item()/len(p)
out=[]
for r in rows:
    if r['collapse_status']!='collapsed': continue
    s,seed,subj=int(r['session']),int(r['seed']),int(r['test_subject'])
    ck=torch.load(Path(r['path'].replace('results/','checkpoints/',1)).with_suffix('.pt'),map_location='cpu',weights_only=False)
    m=ReferenceSCVAE(); m.load_state_dict(ck['model_state_dict'])
    sx,sy,tx,ty=load_fold(subj,s,DEFAULT_DATA_DIR)
    a0,sh0=ev(m,tx,ty); sa0,_=ev(m,sx,sy)
    torch.manual_seed(seed)
    sl=DataLoader(TensorDataset(sx),batch_size=128,shuffle=True); tl=DataLoader(TensorDataset(tx),batch_size=128,shuffle=False)
    steps=min(len(sl),len(tl))
    m.train()
    with torch.no_grad():
        for ep in range(5):
            si=iter(sl); ti=cycle(tl)
            for _ in range(steps):
                m(next(si)[0]); m(next(ti)[0])
    a1,sh1=ev(m,tx,ty); sa1,_=ev(m,sx,sy)
    print(f"s{s} seed{seed} subj{subj:2d} | saved: tgt {a0:.3f} maxshare {sh0:.2f} srcacc {sa0:.3f} | replayed: tgt {a1:.3f} maxshare {sh1:.2f} srcacc {sa1:.3f}",flush=True)
    out.append((a0,a1,sa0,sa1))
o=np.array(out); print("mean saved tgt %.3f -> replayed %.3f | src %.3f -> %.3f"%tuple(o.mean(0)[[0,1,2,3]]))
