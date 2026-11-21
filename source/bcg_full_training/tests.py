"""Targeted checks for full-cohort indexing, RR gaps, gradient flow and saved-report recovery."""
import argparse
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from common import config,NAMES,save,relative_features,rr_inputs,report
from sequences import SequenceDataset,ChunkSampler
import temporal
from temporal import TemporalCNN,evaluate


def fixture(root):
    cfg=config();cfg.update(loader_workers=0,batch_size=2,max_epochs=1,patience=1)
    ids={'train':['0001'],'validation':['0002'],'test':['0003']};(root/'records').mkdir(parents=True)
    counts={};rng=np.random.default_rng(73)
    for part,group in ids.items():
        n=130;y=np.arange(n)%5;common=np.arange(n)%3!=0;expanded=np.arange(n)%11!=0;expanded|=common
        x=rng.normal(size=(4,n,len(NAMES))).astype('float32');x[:,::9,:5]=np.nan
        rr=rng.normal(size=(4,n,3,60)).astype('float32');rr[:,:,2]=1
        np.savez_compressed(root/'records'/f'{part}_{group[0]}.npz',x=x,rr=rr,y=y,common=common,expanded=expanded)
        counts[part]={'common':int(common.sum()),'expanded':int(expanded.sum()),'eligible':n}
    save(root/'manifest.json',{'status':'COMPLETE','feature_names':NAMES,'config':cfg,'participants':ids,
                              'feature_version':'fixture','counts':counts})
    np.savez(root/'population_scaler.npz',center=np.zeros(len(NAMES)),scale=np.ones(len(NAMES)))
    return cfg


def run():
    torch.set_num_threads(2)
    raw=np.array([[1,np.nan],[3,2],[5,4]],float);out,center,scale=relative_features(raw)
    assert out.shape==(3,4) and np.allclose(center,[3,3]) and np.isnan(out[0,3])
    cfg=config();peaks=np.r_[np.arange(1,51),np.arange(61,120)].astype(float)
    normal=np.ones(len(peaks)-1,bool);rr,local=rr_inputs(peaks,normal,4,np.ones(4,bool),cfg,peaks[1:])
    assert rr.shape==(4,3,60) and local.shape==(4,16)
    assert not rr.reshape(4,3,60)[1,2,42:58].any(),'No interpolation across long gap'
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/'prepared';cfg=fixture(root)
        ds=SequenceDataset(root,'train','expanded')
        for training in (False,True):
            requests=list(ChunkSampler(ds,training=training,epoch=3))
            assert len(requests)==len(ds) and len(set((i,s) for i,s,k in requests))==len(ds)
            target_ids=[]
            for request in requests:
                feature,rr,target,common,expanded,i,start=ds[request]
                assert feature.shape==(124,280) and rr.shape==(124,3,60)
                target_ids.extend((int(i),int(start+j)) for j in np.flatnonzero(target>=0))
            assert len(target_ids)==len(set(target_ids))==int(ds.counts.sum())
        feat,rr,*_=ds[(0,0,0)];assert np.all(feat[:30,140:]==1) and np.all(rr[:30]==0)
        for branch in (False,True):
            model=TemporalCNN(cfg,branch);z=model(torch.tensor(feat)[None],torch.tensor(rr)[None])
            assert z.shape==(1,64,5);z.sum().backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        pred=Path(tmp)/'pred.csv';r=evaluate(model,ds,cfg,predictions_path=pred,workers=0)
        assert r['common']['n_epochs']==int(ds.common_counts.sum()) and r['expanded']['n_epochs']==int(ds.counts.sum())
        from finalize import prediction_report
        common,ids,_=prediction_report(pred,True);assert common['n_epochs']==int(ds.common_counts.sum())
        ablated=SequenceDataset(root,'train','expanded',drop_features=[0,70],drop_rr=True)
        f,r,*_=ablated[(0,0,0)];assert np.all(f[:,[0,70]]==0) and np.all(f[:,[140,210]]==1) and not r.any()
        temporal.config=lambda:cfg
        args=SimpleNamespace(prepared=root,output=Path(tmp)/'trained',variant='tcn_dual_expanded',threads=2,device='cpu')
        temporal.train(args);temporal.train(args)
        assert (args.output/'result.json').exists()
        # Fit APIs and weighting are exercised without consuming real validation/test.
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        x=np.random.default_rng(1).normal(size=(100,10));y=np.arange(100)%5;weights=np.ones(100)
        pipe=make_pipeline(SimpleImputer(add_indicator=True),RandomForestClassifier(n_estimators=3,n_jobs=2))
        pipe.fit(x,y,randomforestclassifier__sample_weight=weights)
        tree=XGBClassifier(n_estimators=3,n_jobs=2,num_class=5,objective='multi:softprob')
        tree.fit(x,y,sample_weight=weights,eval_set=[(x,y)],verbose=False)
        assert tree.predict_proba(x,iteration_range=(0,2)).shape==(100,5)
        # Report regeneration is read-only with respect to fitting and inference.
        from full_report import build_report
        run=Path(tmp)/'report';run.mkdir();val={}
        for i in range(10):val['fixture_'+str(i)]={'selection_score':common['macro_f1'],'validation':{str(n):common for n in (0,20,40,60)},'training_epochs':100}
        save(run/'validation_results.json',val);save(run/'selection.json',{'winner':'fixture_0'})
        save(run/'test_results.json',{'five_stage':{str(n):common for n in (0,20,40,60)},'expanded':{}})
        save(run/'preparation_manifest.json',{'counts':ds.manifest['counts']})
        save(run/'feature_importance_validation.json',[{'feature':'fixture','mean_macro_f1_drop':.1,'std':0.}])
        save(run/'feature_importance_method.json',{'method':'Synthetic fixture'})
        save(run/'coverage.json',[{'partition':'train','realized_mae_ms':{str(n):n for n in (0,20,40,60)}}])
        _,_,rows=prediction_report(pred,True)
        for n in (0,20,40,60):
            with (run/f'test_predictions_{n}ms.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        build_report(run);assert (run/'results_bundle.zip').exists()
    print('PASS: RR gaps, normalization, exact-once targets, boundaries, gradients, complete validation, checkpoint/skip, tree APIs, report rendering')

if __name__=='__main__':run()
