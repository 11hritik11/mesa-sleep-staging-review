"""Locked model inference with recoverable test prediction caches and no test selection."""
import csv,hashlib
from pathlib import Path
import joblib,numpy as np,torch
from balanced_utils import (cfg,base,oldconfig,feature_names,probability_transform,matrix,TemporalCNN)
from balanced_data import BalancedDataset
from balanced_models import BalancedNetwork
from train_balanced_deep import validation_logits
from train_balanced_classic import raw_scores

class Predictor:
 def __init__(self,prepared,results,cache=None):self.prepared,self.results,self.cache=prepared,results,cache;self.models={}
 def predict(self,name,part,level,expanded=False,drop_features=(),drop_rr=False):
  info=self.results[name];cache_path=None
  if part=='test' and self.cache is not None:
   self.cache.mkdir(exist_ok=True,parents=True);cache_path=self.cache/f'{name}_{level}ms.npz'
   if cache_path.exists():
    with np.load(cache_path) as f:
     if str(f['source'])!=info['source']:raise ValueError('Test cache provenance changed')
     return f['prob'],f['y'],f['pids'],f['epochs']
  if info['kind']=='ensemble':
   members=[self.predict(member,part,level,expanded,drop_features,drop_rr) for member in info['members']]
   for p,y,pid,e in members[1:]:
    if not all(np.array_equal(a,b) for a,b in zip((y,pid,e),members[0][1:])):raise ValueError('Ensemble identities differ')
   scores=np.log(np.maximum(np.mean([v[0] for v in members],axis=0),1e-12));y,pids,epochs=members[0][1:]
  elif info['kind']=='control' and info['variant']=='tcn_dual_common' and part=='test':
   # Reuse the prior completed test inference rather than running the same checkpoint again.
   path=Path(cfg()['base_run'])/'results'/f'expanded_test_predictions_{level}ms.csv'
   with path.open() as f:rows=list(csv.DictReader(f))
   scores=np.log(np.maximum(np.array([[float(r['p_'+s]) for s in ['Wake','N1','N2','N3','REM']] for r in rows]),1e-12))
   y=np.array([int(r['reference']) for r in rows]);pids=np.array([r['participant'] for r in rows]);epochs=np.array([int(r['epoch']) for r in rows])
  elif info['kind']=='classic':
   if name not in self.models:self.models[name]=joblib.load(Path(info['output'])/'model.joblib')
   x,y,pids,epochs=matrix(self.prepared,part,level,modality=info['specification']['modality'],mask_name='expanded' if expanded else 'common')
   if drop_features:x[:,drop_features]=np.nan
   scores=raw_scores(self.models[name],x)
  else:
   spec=info['specification'];settings=oldconfig();settings.update(batch_size=cfg()['batch_size'],loader_workers=cfg()['loader_workers'])
   if name not in self.models:
    if info['kind']=='control':
     model=TemporalCNN(oldconfig(),rr_branch=info['variant']!='tcn_hrv_common');checkpoint_path=Path(info['checkpoint'])
    else:
     model=BalancedNetwork(len(feature_names(spec['modality'])),spec['architecture'],settings);checkpoint_path=Path(info['output'])/'best.pt'
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);self.models[name]=model
   ds=BalancedDataset(self.prepared,part,'expanded',spec['modality']);ds.drop_features=tuple(drop_features);ds.drop_rr=drop_rr
   scores,y,pids,epochs=validation_logits(self.models[name],ds,settings,[0,20,40,60].index(level),collect_expanded=expanded)
  prob=probability_transform(scores,np.array(info['prior']),info['beta'])
  if cache_path is not None:
   temporary=cache_path.with_suffix('.tmp.npz');np.savez_compressed(temporary,prob=prob,y=y,pids=pids,epochs=epochs,source=info['source']);temporary.replace(cache_path)
  return prob,y,pids,epochs
