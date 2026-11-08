#!/usr/bin/env python3
"""Dump lightweight pred cache from same pool as foldpure_25var, then run v2 selectors."""
from __future__ import annotations
import json, sys, warnings
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import cv2, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path("/home/chocheng9/kaggle_UMUD")
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts/chase018"))
from sklearn.linear_model import Ridge
from adapt_multexp_fasc import FROZEN, VARIANTS, adapt_fasc, image_features, loocv_cal_pa_mt, predict_one, apo_belly_mask, structure_ridge
from umud.rl.bandit_policy import BanditConfig, SoftmaxBandit
HK=timezone(timedelta(hours=8)); PRIOR_C=0.13195144580335014; GATE=0.12
OUT=ROOT/"reports/phase3/chase018"; CACHE=OUT/"cache"/"foldpure_25var_predcache.npz"
def adapt_or_blend(g,apo,fasc,variant):
    if variant.startswith("blend_bm_"):
        w=float(variant.split("_")[-1])/100.0
        return np.clip((1-w)*fasc+w*adapt_fasc(g,apo,fasc,"morph_connect"),0,1).astype(np.float32)
    if variant.startswith("blend_br_"):
        w=float(variant.split("_")[-1])/100.0
        return np.clip((1-w)*fasc+w*adapt_fasc(g,apo,fasc,"ridge_blend03"),0,1).astype(np.float32)
    if variant.startswith("blend_bc_"):
        w=float(variant.split("_")[-1])/100.0
        return np.clip((1-w)*fasc+w*adapt_fasc(g,apo,fasc,"corridor_dilate"),0,1).astype(np.float32)
    return adapt_fasc(g,apo,fasc,variant).astype(np.float32)
POOL=list(VARIANTS)+[f"blend_bm_{w}" for w in (15,25,40,50,65,75)]+[f"blend_br_{w}" for w in (15,25,40,50)]+[f"blend_bc_{w}" for w in (15,25,40,50)]
REDUCED=["baseline","ridge_blend03","ridge_blend05","ridge_replace","morph_prune","morph_connect","frangi_like","belly_orient15","belly_orient25","corridor_dilate","blend_br_15","blend_br_25","blend_br_40","blend_br_50"]
SAFE=["baseline","blend_br_40","morph_prune","ridge_replace","ridge_blend03","blend_br_15"]
def raw_err(pa,fl,gt_pa,gt_fl): return abs(pa-gt_pa)/6.0+abs(fl-gt_fl)/12.0
def rich_feats(g,apo,fasc,px,pas,fls,pool):
    base=image_features(g,apo,fasc,px); h,w=apo.shape; belly=apo_belly_mask(apo,0.60)
    g_r=cv2.resize(g,(w,h),interpolation=cv2.INTER_AREA); ridge,ang,coher=structure_ridge(g_r,sigma=1.5); bi=pool.index("baseline")
    out=dict(base); out.update(fasc_hi=float((fasc>0.5).mean()),fasc_lo=float((fasc>0.2).mean()),ridge_belly=float(ridge[belly>0].mean()) if belly.sum() else 0.0,pa_std=float(np.std(pas)),fl_std=float(np.std(fls)),pa_iqr=float(np.percentile(pas,75)-np.percentile(pas,25)),fl_iqr=float(np.percentile(fls,75)-np.percentile(fls,25)),pa_range=float(pas.max()-pas.min()),fl_range=float(fls.max()-fls.min()),base_pa=float(pas[bi]),base_fl=float(fls[bi]),thr_slope=float((fasc>0.15).mean()-(fasc>0.55).mean()),q_scale=float(px)); return out
def build_cache():
    expert=pd.read_csv(ROOT/"data/external/umud_expert_arch_v0.1.0/expert_consensus.csv"); expert_dir=ROOT/"data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"; maps=OUT/"cache"/"maps_multexp"; kw=dict(FROZEN)
    names,gpa,gfl,gmt,feat_list=[],[],[],[],[]; PA=np.zeros((len(expert),len(POOL))); FL=np.zeros_like(PA); MT=np.zeros_like(PA)
    for i,(_,er) in enumerate(expert.iterrows()):
        name=er["image_file"]; g=cv2.imread(str(expert_dir/name),cv2.IMREAD_GRAYSCALE); z=np.load(maps/f"{name}.npz"); apo,fasc=z["apo"].astype(np.float32),z["fasc"].astype(np.float32); px=float(er["Scale_pixel_per_cm"])/10.0
        for j,v in enumerate(POOL):
            m=adapt_or_blend(g,apo,fasc,v); pr=predict_one(g,apo,m,px,kw); PA[i,j],FL[i,j],MT[i,j]=pr["pred_pa"],pr["pred_fl"],pr["pred_mt"]
        feat_list.append(rich_feats(g,apo,fasc,px,PA[i],FL[i],POOL)); names.append(name); gpa.append(float(er["median_PA"])); gfl.append(float(er["median_FL"])); gmt.append(float(er["median_MT"])); print("cached",name,flush=True)
    keys=sorted(feat_list[0].keys()); X=np.array([[float(f[k]) if np.isfinite(f[k]) else np.nan for k in keys] for f in feat_list]); med=np.nanmedian(X,0); bad=~np.isfinite(X); X[bad]=np.take(med,np.where(bad)[1])
    CACHE.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(CACHE,PA=PA,FL=FL,MT=MT,X=X,keys=np.array(keys),names=np.array(names),gpa=np.array(gpa),gfl=np.array(gfl),gmt=np.array(gmt),pool=np.array(POOL)); print("wrote",CACHE,flush=True)
def costs(PA,FL,gpa,gfl,cols):
    n=PA.shape[0]; C=np.zeros((n,len(cols)))
    for i in range(n):
        for k,j in enumerate(cols): C[i,k]=raw_err(PA[i,j],FL[i,j],gpa[i],gfl[i])
    return C
def bandit_pick(X,C,cols_global,epochs_extra=0):
    n,V=C.shape; cfgs=[dict(lr=0.05,l2=0.1,epochs=50,temperature=1.0),dict(lr=0.08,l2=0.2,epochs=60,temperature=0.8),dict(lr=0.04,l2=0.3,epochs=80,temperature=1.2),dict(lr=0.1,l2=0.15,epochs=40,temperature=0.7)]; picks=np.zeros(n,dtype=int)
    for i in range(n):
        tr=[j for j in range(n) if j!=i]; mu,sd=X[tr].mean(0),X[tr].std(0)+1e-6; Xtr=(X[tr]-mu)/sd; xt=(X[i]-mu)/sd; best_sc,best_band=1e18,None
        for cfg in cfgs:
            band=SoftmaxBandit(BanditConfig(n_actions=V,state_dim=X.shape[1],lr=cfg["lr"],l2=cfg["l2"],epochs=cfg["epochs"]+epochs_extra,seed=0,temperature=cfg["temperature"])); rng=np.random.default_rng(0)
            for ep in range(cfg["epochs"]+epochs_extra):
                for u in rng.permutation(len(tr)):
                    s=Xtr[u]; a=band.act(s,greedy=False,rng=rng); r=-C[tr[u],a]; band.update(s,a,r-(-float(C[tr[u]].mean()))); band.update(s,int(np.argmin(C[tr[u]])),0.3)
            sc=float(np.mean([C[tr[u],band.act(Xtr[u],greedy=True)] for u in range(len(tr))]))
            if sc<best_sc: best_sc,best_band=sc,band
        picks[i]=cols_global[int(best_band.act(xt,greedy=True))]
    return picks
def feature_rule_pick(X,C,cols_global,top_m=2):
    n=X.shape[0]; ora=C.argmin(1); picks=np.zeros(n,dtype=int)
    for i in range(n):
        tr=[j for j in range(n) if j!=i]; top=[v for v,_ in Counter(ora[j] for j in tr).most_common(top_m)]
        if len(top)==1: picks[i]=cols_global[top[0]]; continue
        v0,v1=top[0],top[1]; best=(cols_global[v0],1e18)
        for fi in range(X.shape[1]):
            vals=np.unique(X[tr,fi])
            if len(vals)<2: continue
            thrs=0.5*(vals[:-1]+vals[1:])
            if len(thrs)>25: thrs=thrs[::max(1,len(thrs)//25)]
            for thr in thrs:
                for lo_is_v0 in (True,False):
                    cs=[C[j,v0 if ((X[j,fi]<thr) if lo_is_v0 else (X[j,fi]>=thr)) else v1] for j in tr]; sc=float(np.mean(cs))
                    if sc<best[1]:
                        use0=(X[i,fi]<thr) if lo_is_v0 else (X[i,fi]>=thr); best=(cols_global[v0 if use0 else v1],sc)
        picks[i]=best[0]
    return picks
def trainS_pick(C,cols_global):
    n=C.shape[0]; picks=np.zeros(n,dtype=int)
    for i in range(n):
        m=np.ones(n,dtype=bool); m[i]=False; picks[i]=cols_global[int(np.argmin(C[m].mean(0)))]
    return picks
def residual_pick(X,C,cols_global):
    n,V=C.shape; alphas=[0.3,1.0,3.0,10.0,30.0]; picks=np.zeros(n,dtype=int)
    for i in range(n):
        tr=np.array([j for j in range(n) if j!=i]); mu,sd=X[tr].mean(0),X[tr].std(0)+1e-6; Xtr=(X[tr]-mu)/sd; xt=(X[i]-mu)/sd; best_a,best_sc=alphas[0],1e18
        folds=np.array_split(np.arange(len(tr)),5)
        for a in alphas:
            errs=[]
            for fi,val in enumerate(folds):
                tin=np.concatenate([folds[fj] for fj in range(5) if fj!=fi]); pred=np.zeros((len(val),V))
                for v in range(V):
                    r=Ridge(alpha=a); r.fit(Xtr[tin],C[tr[tin],v]); pred[:,v]=r.predict(Xtr[val])
                for local_k,row in zip(val,pred): errs.append(C[tr[local_k],int(np.argmin(row))])
            sc=float(np.mean(errs))
            if sc<best_sc: best_sc,best_a=sc,a
        pred_c=np.zeros(V)
        for v in range(V):
            r=Ridge(alpha=best_a); r.fit(Xtr,C[tr,v]); pred_c[v]=r.predict(xt.reshape(1,-1))[0]
        picks[i]=cols_global[int(np.argmin(pred_c))]
    return picks
def main():
    if not CACHE.exists(): build_cache()
    z=np.load(CACHE,allow_pickle=True); PA,FL,MT,X=z["PA"],z["FL"],z["MT"],z["X"]; gpa,gfl,gmt=z["gpa"],z["gfl"],z["gmt"]; names=list(z["names"]); pool=list(z["pool"]); n=len(names)
    results={}; best=None
    def run(name,picks):
        nonlocal best
        rows,pnames=[],[]
        for i in range(n):
            j=int(picks[i]); pnames.append(pool[j]); rows.append(dict(image_file=names[i],ok=True,pred_pa=float(PA[i,j]),pred_fl=float(FL[i,j]),pred_mt=float(MT[i,j]),gt_pa=float(gpa[i]),gt_fl=float(gfl[i]),gt_mt=float(gmt[i])))
        df=pd.DataFrame(rows); comps,mae,_,_=loocv_cal_pa_mt(df); full_C=costs(PA,FL,gpa,gfl,list(range(len(pool)))); ora=full_C.argmin(1)
        hit1=sum(1 for i in range(n) if picks[i]==ora[i])/n; hit3=sum(1 for i in range(n) if picks[i] in set(np.argsort(full_C[i])[:3]))/n
        r=dict(name=name,S=float(comps["S"]),MAE=mae,gate=float(comps["S"])<GATE,beats_C=float(comps["S"])<PRIOR_C,pick_counts=dict(Counter(pnames)),acc_at_1=hit1,acc_at_3=hit3)
        results[name]=r; print(f"{name}: S={r['S']:.5f} beats_C={r['beats_C']} acc@1={hit1:.3f}",flush=True)
        if best is None or r["S"]<best["S"]: best={**r,"picks":pnames,"df":df}
        return r
    full_C=costs(PA,FL,gpa,gfl,list(range(len(pool)))); run("leaky_oracle",full_C.argmin(1))
    run("nested_trainS_full",trainS_pick(full_C,list(range(len(pool)))))
    print("=== full bandit imitation ===",flush=True); run("nested_bandit_imitation_full",bandit_pick(X,full_C,list(range(len(pool)))))
    run("nested_feature_rule_full",feature_rule_pick(X,full_C,list(range(len(pool))),2))
    run("nested_residual_full",residual_pick(X,full_C,list(range(len(pool)))))
    cols=[pool.index(v) for v in REDUCED]; Cred=costs(PA,FL,gpa,gfl,cols)
    print("=== reduced ===",flush=True)
    run("leaky_oracle_reduced",np.array([cols[int(o)] for o in Cred.argmin(1)]))
    run("nested_trainS_reduced",trainS_pick(Cred,cols))
    run("nested_bandit_imitation_reduced",bandit_pick(X,Cred,cols))
    run("nested_bandit_imitation_reduced_long",bandit_pick(X,Cred,cols,epochs_extra=40))
    run("nested_feature_rule_reduced",feature_rule_pick(X,Cred,cols,2))
    run("nested_residual_reduced",residual_pick(X,Cred,cols))
    cols_s=[pool.index(v) for v in SAFE]; Cs=costs(PA,FL,gpa,gfl,cols_s)
    print("=== safe ===",flush=True)
    run("leaky_oracle_safe",np.array([cols_s[int(o)] for o in Cs.argmin(1)]))
    run("nested_trainS_safe",trainS_pick(Cs,cols_s)); run("nested_bandit_safe",bandit_pick(X,Cs,cols_s))
    run("nested_feature_rule_safe",feature_rule_pick(X,Cs,cols_s,2)); run("nested_residual_safe",residual_pick(X,Cs,cols_s))
    br=feature_rule_pick(X,Cred,cols,2); bb=bandit_pick(X,Cred,cols); ts=trainS_pick(Cred,cols); base_i=pool.index("baseline")
    run("nested_hybrid_agree_else_baseline",np.array([br[i] if br[i]==bb[i] else base_i for i in range(n)]))
    run("nested_hybrid_agree_else_trainS",np.array([br[i] if br[i]==bb[i] else ts[i] for i in range(n)]))
    leak=results["leaky_oracle"]; cand={k:v for k,v in results.items() if not k.startswith("leaky")}
    best_name=min(cand,key=lambda k:cand[k]["S"]); best_r=cand[best_name]
    promote=bool(best_r["S"]<PRIOR_C); gate=bool(best_r["S"]<GATE); gap=float(best_r["S"]-leak["S"])
    payload=dict(asof=datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong"),tag="foldpure_25var_v2",results=results,
        best_transferable={k:best_r[k] for k in best_r if k not in ("df","picks")},best_name=best_name,leaky_oracle=leak,
        oracle_selector_gap=gap,prior_C=PRIOR_C,delta_vs_C=float(best_r["S"]-PRIOR_C),promote=promote,kaggle_submit=False,
        reduced_pool=REDUCED,safe_pool=SAFE,prior_v1_best=0.1332811760145091)
    payload["gate_0.12"]=gate
    payload["oracle_selector_gap"]=gap
    prev_path=OUT/"multexp_foldpure_25var_scores.json"; prev=json.loads(prev_path.read_text()) if prev_path.exists() else {}
    prev["v2"]=payload; prev["asof"]=payload["asof"]
    if best_r["S"]<prev.get("best_transferable",{}).get("S",1e9):
        prev["best_transferable"]=payload["best_transferable"]; prev.setdefault("results",{}).update({f"v2_{k}":v for k,v in results.items()})
    prev_path.write_text(json.dumps(prev,indent=2,default=str))
    (OUT/"multexp_foldpure_25var_summary.json").write_text(json.dumps(dict(best_S=best_r["S"],best_name=best_name,MAE=best_r["MAE"],oracle_S=leak["S"],gap=gap,prior_C=PRIOR_C,gate=gate,promote=promote,kaggle_submit=False,v1_best=0.1332811760145091),indent=2))
    if "df" in best: best["df"].to_csv(OUT/"multexp_foldpure_25var_best_rows.csv",index=False)
    lines=[f"# MultExp fold-pure 25-var selector — {payload['asof']}","",
        "## Result (honest nested LOOCV)",f"- **Best transferable S = {best_r['S']:.5f}** (`{best_name}`)",
        f"  PA {best_r['MAE']['pa_deg']:.3f}° / FL {best_r['MAE']['fl_mm']:.3f} mm / MT {best_r['MAE']['mt_mm']:.3f} mm",
        f"- vs prior C **0.13195** Δ {payload['delta_vs_C']:+.5f}",f"- vs v1 bandit 0.13328 Δ {best_r['S']-0.1332811760145091:+.5f}",
        f"- Leaky oracle S = **{leak['S']:.5f}** | gap = **{gap:.5f}**",
        f"- Acc@1/@3 = {best_r.get('acc_at_1', float('nan')):.3f} / {best_r.get('acc_at_3', float('nan')):.3f}",
        f"- **Gate 0.12: {'PASS' if gate else 'FAIL'}** | Promote: {promote} | **No Kaggle**","","## Top methods","| Method | S | beats C |","|--------|---|---------|"]
    for k,v in sorted(cand.items(), key=lambda kv: kv[1]["S"])[:15]: lines.append(f"| {k} | {v['S']:.5f} | {v['beats_C']} |")
    lines += [f"| leaky_oracle | {leak['S']:.5f} | yes |","","## Diagnosis",f"Best transferable {best_r['S']:.5f}; oracle {leak['S']:.5f}; gap {gap:.5f}.",
        "Selectors cannot close oracle−selector gap on Frundles+A 25-var pool." if best_r["S"]>=0.13 else ("Improved but gate still open." if not gate else "Gate cleared."),
        "","## Next lever"]
    if best_r["S"]>=0.13: lines.append("Transferable S stays ≥0.13 despite oracle ~0.105 — get fasc maps outside Frundles+A; do not spend more on fold-pure selectors for this pool.")
    elif not gate: lines.append(f"Beats C but above 0.12 (gap {best_r['S']-GATE:.4f}); keep recipe, need better maps.")
    else: lines.append("Gate cleared — await parent approval before Kaggle.")
    (OUT/"multexp_foldpure_25var_status.md").write_text("\n".join(lines)+"\n")
    if promote:
        print("PROMOTE rebuilding maps...",flush=True)
        expert=pd.read_csv(ROOT/"data/external/umud_expert_arch_v0.1.0/expert_consensus.csv"); expert_dir=ROOT/"data/external/umud_expert_arch_v0.1.0/benchmark_dataset_architecture_v0.1.0"
        maps_in=OUT/"cache"/"maps_multexp"; maps_out=OUT/"cache"/"maps_multexp_foldpure_25var"; maps_out.mkdir(parents=True,exist_ok=True)
        pick_by_name=dict(zip(names,best["picks"]))
        for _,er in expert.iterrows():
            name=er["image_file"]; g=cv2.imread(str(expert_dir/name),cv2.IMREAD_GRAYSCALE); z0=np.load(maps_in/f"{name}.npz"); apo,fasc0=z0["apo"].astype(np.float32),z0["fasc"].astype(np.float32)
            fasc=adapt_or_blend(g,apo,fasc0,pick_by_name[name]); np.savez_compressed(maps_out/f"{name}.npz",apo=apo.astype(np.float16),fasc=fasc.astype(np.float16))
        recipe=dict(asof=payload["asof"],name=best_name,S_loocv_cal_pa_mt=best_r["S"],MAE_loocv=best_r["MAE"],kw={k:FROZEN[k] for k in ("fasc_thr","apo_thr","tex_w","fl_factor","fl_sin_w","agree_thr","pa_mode")},fasc_adapt=dict(type=best_name,pick_counts=best_r["pick_counts"],maps=str(maps_out)),cache_dir=str(maps_out),prior_C_S=PRIOR_C,beats_prior_C=True,kaggle_submit=False,lab_fasc_ckpt=False)
        recipe["gate_0.12"]=gate
        recipe["gap_to_0.12"]=float(best_r["S"]-GATE)
        (OUT/"math_map_promoted_recipe.json").write_text(json.dumps(recipe,indent=2)); print("PROMOTED",flush=True)
    print(f"V2_RESULT S={best_r['S']:.5f} name={best_name} leaky={leak['S']:.5f} gap={gap:.5f} vs_C={PRIOR_C:.5f} gate={gate} promote={promote}",flush=True)
if __name__=="__main__": main()