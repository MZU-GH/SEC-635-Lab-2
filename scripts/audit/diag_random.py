# usage: diag_random.py HTTP_RESULT.json [STORED.json]  (in-process random search x2 vs stored vs HTTP)
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys, json, numpy as np, joblib
sys.path.insert(0,str(ROOT))
from detector.dataset import build_split
from detector.evasion import *
sp=build_split(); m=joblib.load(str(ROOT)+"/lab1/outputs/models/task1_wednesday_mlp_neuralnet_temporal.joblib")
X=evaluation_flows(sp,m,n=300); mut=classify(sp.feature_names); scale=m.named_steps["scaler"].scale_
stored=json.load(open(sys.argv[2] if len(sys.argv)>2 else str(ROOT)+"/results/task1b_evasion_wednesday.json"))["curve"]
http=json.load(open(sys.argv[1]))["curve"]
def run():
    rng=np.random.default_rng(SEED); sc=Scorer(m,sp); out={}
    for eps in EPS_GRID[1:]:
        rad=raw_radius(scale,eps,mut); ev=0; calls=0
        for xi in X:
            xa,b=attack_random(sc,xi,rad,rng,tries=200); calls+=b.model_calls
            if m.predict_proba(xa[None,:])[0][1]<DETECT_THRESHOLD: ev+=1
        out[eps]=(1-ev/len(X),calls); print(eps,out[eps],flush=True)
    return out
r1=run(); r2=run()
print("eps | stored | http | run1 | run2")
for c,h in zip(stored[1:],http[1:]):
    e=c['eps_std']; a=c['attacks']['random_search']; b=h['attacks']['random_search']
    print(e,(a['detection_rate'],a['budget']['model_calls']),(b['detection_rate'],b['budget']['model_calls']),r1[e],r2[e])
print("run1==run2",r1==r2,"run1==stored",all(r1[c['eps_std']]==(c['attacks']['random_search']['detection_rate'],c['attacks']['random_search']['budget']['model_calls']) for c in stored[1:]))
