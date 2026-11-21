"""Compare saved classifier BN running stats with source/target feature stats on collapsed and healthy folds.

Run: python -m src.experiments.check_bn_running_stats
"""
import csv, torch
from pathlib import Path
from src.models.scvae_reference import ReferenceSCVAE
from src.experiments.train_scvae_reference import load_fold, DEFAULT_DATA_DIR
rows=[x for x in csv.DictReader(open('results/diagnostics/collapse/reference_4s/folds.csv'))]
pick=[x for x in rows if x['collapse_status']=='collapsed']+[x for x in rows if x['collapse_status']=='healthy'][:5]
def d(a,b,var): # relative distance
    return ((a-b).norm()/b.norm()).item()
print("status  s seed subj | dist(run_mean, src) dist(run_mean, tgt_all) dist(run_mean, tgt_last) | var: src tgt_all tgt_last")
for x in pick:
    s,seed,subj=int(x['session']),int(x['seed']),int(x['test_subject'])
    ck=torch.load(Path(x['path'].replace('results/','checkpoints/',1)).with_suffix('.pt'),map_location='cpu',weights_only=False)
    m=ReferenceSCVAE(); m.load_state_dict(ck['model_state_dict']); m.eval()
    bn=m.classifier[1]; rm,rv=bn.running_mean,bn.running_var
    sx,_,tx,_=load_fold(subj,s,DEFAULT_DATA_DIR)
    with torch.no_grad():
        f=lambda X: m.classifier[0](m.encode(X)[0])
        hs,ht=f(sx),f(tx)
        n=len(tx); last=hs.new_tensor(0)
        # last target batch of each epoch (unshuffled, batch 128, 7 steps)
        steps=min(-(-len(sx)//128), -(-n//128)); lb=ht[(steps-1)*128: steps*128]
    a=torch.linspace(0,1,21); best=min(d(rm,(1-t)*hs.mean(0)+t*ht.mean(0),0) for t in a)
    print(f"src-vs-tgt {d(hs.mean(0),ht.mean(0),0):.2f} best-mixture {best:.2f} ",end="")
    print(f"{x['collapse_status']:8s}{s} {seed:3d} {subj:3d}  | {d(rm,hs.mean(0),0):.2f} {d(rm,ht.mean(0),0):.2f} {d(rm,lb.mean(0),0):.2f} | {d(rv,hs.var(0),0):.2f} {d(rv,ht.var(0),0):.2f} {d(rv,lb.var(0),0):.2f}")
