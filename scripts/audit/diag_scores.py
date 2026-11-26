# usage: diag_scores.py BASE_URL   (API serving the Lab 1 Wednesday model)
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys, numpy as np, joblib, requests
sys.path.insert(0,str(ROOT))
from detector.dataset import build_split
from detector.evasion import evaluation_flows, raw_radius
from detector.features import classify, project_feasible
sp=build_split(); m=joblib.load(str(ROOT)+"/lab1/outputs/models/task1_wednesday_mlp_neuralnet_temporal.joblib")
X=evaluation_flows(sp,m,n=300); mut=classify(sp.feature_names); rad=raw_radius(m.named_steps["scaler"].scale_,0.5,mut)
rng=np.random.default_rng(1)
C=np.array([project_feasible(np.clip(x+rng.uniform(-1,1,x.shape)*rad,x-rad,x+rad)[None],sp.feature_names)[0] for x in X[:100] for _ in range(10)])
U=sys.argv[1]+"/score_batch"  # base URL of an API serving the Lab 1 Wednesday model
ip1=np.array([m.predict_proba(c[None])[0][1] for c in C])
ipb=m.predict_proba(C)[:,1]
h1=np.array([requests.post(U,json={"features":c[None].tolist()}).json()["scores"][0] for c in C])
hb=np.array(requests.post(U,json={"features":C.tolist()}).json()["scores"])
print("n",len(C))
print("max|inproc single - inproc batch|",np.abs(ip1-ipb).max())
print("max|inproc single - http single|",np.abs(ip1-h1).max())
print("max|inproc batch - http batch|",np.abs(ipb-hb).max())
print("verdict flips single inproc vs http single",int(((ip1<.5)!=(h1<.5)).sum()))
print("nonfinite in C",int((~np.isfinite(C)).sum()))
