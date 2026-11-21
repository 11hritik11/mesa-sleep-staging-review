"""Synthetic all-family integration and two real-record extraction checks; no test tuning."""
import argparse,csv,json,tempfile,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from balanced_utils import (ROOT,cfg,base,save,read_json,fingerprint,EXTRA_NAMES,feature_names,
                           report,ba_from_cm,CLASSIC_SPECS,DEEP_SPECS,choose_beta_across_levels)
from prepare_balanced import respiratory_features,extract


def run(real=False):
 torch.set_num_threads(2);started_version=fingerprint()
 if real:
  original=read_json(base()/'manifest.json');out=ROOT/'data/derived/bcg_balanced_smoke'/started_version[:12];(out/'extra').mkdir(parents=True,exist_ok=True)
  for pid in ('0001','0167'):
   part=next(p for p,people in original['participants'].items() if pid in people)
   row=extract((out,part,pid,fingerprint()));assert row['xml_identity_verified'] and row['clock_checks_pass']
   print('REAL EXTRACTION PASS',pid,row,flush=True)
 hz=4;signal=np.sin(2*np.pi*.25*np.arange(2400)/hz)
 features=respiratory_features(signal,20)
 assert abs(float(features[10,0])-.25)<.04 and 0<=features[10,1]<=1 and features[10,5]==1
 y=np.arange(100)%5;pred=y.copy();pids=np.repeat('fixture',100)
 assert report(y,pred,pids)['balanced_accuracy']==1 and ba_from_cm(np.eye(5))==1
 assert report(np.zeros(10,dtype=int),np.zeros(10,dtype=int),np.repeat('x',10))['balanced_accuracy'] is None
 with tempfile.TemporaryDirectory() as temp:
  temp=Path(temp);baseline=temp/'base'
  from tests import fixture
  oldcfg=fixture(baseline)
  shutil.copyfile(baseline/'records/validation_0002.npz',baseline/'records/validation_0004.npz')
  manifest=read_json(baseline/'manifest.json');manifest['participants']['validation'].append('0004')
  for k in manifest['counts']['validation']:manifest['counts']['validation'][k]*=2
  save(baseline/'manifest.json',manifest)
  prepared=temp/'prepared';(prepared/'extra').mkdir(parents=True)
  rng=np.random.default_rng(9);audits=[]
  for part,people in manifest['participants'].items():
   for pid in people:
    x=rng.normal(size=(130,len(EXTRA_NAMES))).astype('float32');x[::11,:3]=np.nan
    np.savez(prepared/'extra'/f'{part}_{pid}.npz',x=x)
    audits.append({'participant':pid,'partition':part,'channels':{s:True for s in ('Thor','Abdo','Flow','SpO2','Leg')}})
  np.savez(prepared/'extra_scaler.npz',center=np.zeros(len(EXTRA_NAMES)),scale=np.ones(len(EXTRA_NAMES)))
  groups={'calibration':['0002'],'selection':['0004']}
  save(prepared/'balanced_manifest.json',{'status':'COMPLETE','source_version':fingerprint(),'base_feature_version':'fixture','validation_participants':groups})
  save(prepared/'audit.json',audits);save(baseline/'coverage.json',[])
  import balanced_utils as u,balanced_data as d,train_balanced_deep as deep,train_balanced_classic as classic
  original_base=u.base;original_config=u.cfg
  u.base=lambda:baseline;d.base=lambda:baseline;deep.base=lambda:baseline
  settings=cfg();settings.update(epochs=1,patience=1,loader_workers=0,batch_size=2)
  u.cfg=lambda:settings;deep.cfg=lambda:settings;classic.cfg=lambda:settings
  for modality in ('beat','edr','multi'):
   ds=d.BalancedDataset(prepared,'train','expanded',modality)
   f,r,y,_,_,_,_=ds[(0,0,0)];assert f.shape==(124,2*len(feature_names(modality)))
   from balanced_models import BalancedNetwork
   for architecture in ('tcn','gru','lstm','transformer','mlp'):
    model=BalancedNetwork(len(ds.names),architecture,oldcfg);z=model(torch.tensor(f)[None],torch.tensor(r)[None])
    assert z.shape==(1,64,5) and torch.isfinite(z).all();z.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  ds=d.BalancedDataset(prepared,'train','expanded','edr');ds.drop_features=(0,140);ds.drop_rr=True
  f,r,*_=ds[(0,0,0)];assert not r.any() and not f[:,[0,140]].any() and np.all(f[:,[176,316]]==1)
  # Class correction must not change when only selection labels change.
  labels=np.arange(100)%5;persons=np.array(['0002']*50+['0004']*50);scores=rng.normal(size=(100,5));prior=np.array([.4,.1,.3,.05,.15])
  evidence={n:(scores.copy(),labels.copy(),persons.copy()) for n in [0,20,40,60]}
  beta,_,_=choose_beta_across_levels(evidence,prepared,prior)
  changed={n:(z,np.r_[y[:50],(y[50:]+2)%5],p) for n,(z,y,p) in evidence.items()}
  assert choose_beta_across_levels(changed,prepared,prior)[0]==beta
  # Exercise exact fitting APIs on an identifiable fixture with enough rows for leaf-size constraints.
  def classical_fixture(prepared_path,part,level=0,augmented=False,modality='beat',mask_name='common'):
   n=1500;labels=np.arange(n)%5
   xx=np.random.default_rng(15+level).normal(size=(n,len(feature_names(modality)))).astype('float32')
   xx[:,0]=labels*3+xx[:,0]*.1;xx[:,1]=(labels%2)+xx[:,1]*.1
   persons=np.repeat('0001',n) if part=='train' else np.array(['0002']*(n//2)+['0004']*(n//2))
   epochs=np.arange(n) if part=='train' else np.tile(np.arange(n//2),2)
   return xx,labels,persons,epochs
  classic.matrix=classical_fixture
  for index in range(len(CLASSIC_SPECS)):
   args=SimpleNamespace(prepared=prepared,output=temp/'classic',index=index,threads=2)
   classic.train(args)
   result=read_json(args.output/CLASSIC_SPECS[index]['name']/'result.json');assert result['completed']
   import joblib
   loaded=joblib.load(Path(result['output'])/'model.joblib')
   xx,_,_,_=u.matrix(prepared,'validation',modality=result['specification']['modality'])
   assert np.isfinite(classic.raw_scores(loaded,xx)).all()
  for index in range(len(DEEP_SPECS)):
   args=SimpleNamespace(prepared=prepared,output=temp/'deep',index=index,threads=2)
   deep.train(args);deep.train(args)
   assert read_json(args.output/DEEP_SPECS[index]['name']/'result.json')['completed']
  # Test direct saved-output checks and full figure rendering with small fixtures.
  from finalize_balanced import validate_prediction
  rr=report(labels,labels,persons);run=temp/'report';run.mkdir();results={}
  for modality in ('beat','edr','multi'):
   name='fixture_'+modality;results[name]={'name':name,'specification':{'modality':modality},'selection_score':1.,'beta':0.,'history':[]}
  save(run/'validation_results.json',results);save(run/'selection.json',{'tracks':{m:{'winner':'fixture_'+m,'validation_balanced_accuracy':1.} for m in ('beat','edr','multi')}})
  tracks={}
  for modality in ('beat','edr','multi'):
   folder=run/'tracks'/modality;folder.mkdir(parents=True);tracks[modality]={}
   save(folder/'importance_validation.json',[{'group':'fixture','balanced_accuracy_drop':.1}])
   for level in (0,20,40,60):
    path=folder/f'test_predictions_{level}ms.csv'
    with path.open('w') as f:
     w=csv.writer(f);w.writerow(['participant','epoch','reference','prediction','common']+['p_'+s for s in ['Wake','N1','N2','N3','REM']])
     for i,(y,pid) in enumerate(zip(labels,persons)):w.writerow([pid,i,int(y),int(y),1,*np.eye(5)[y]])
    tracks[modality][str(level)]=validate_prediction(path)
  save(run/'test_results.json',tracks)
  import balanced_report as br
  br.base=lambda:baseline;br.build_report(run,prepared)
  assert (run/'results_bundle.zip').exists()
  u.base=original_base;u.cfg=original_config
 if fingerprint()!=started_version:raise ValueError('Scientific sources changed during preflight')
 if real:save(ROOT/'data/derived/bcg_balanced_smoke/PREFLIGHT_PASS.json',{'source':started_version,'status':'PASS'})
 print('PASS: real alignment/extraction, respiratory frequency, balanced metrics, disjoint correction, all architectures, all fitting families, exact targets, checkpoint skip, model serialization and report rendering',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--real',action='store_true');run(p.parse_args().real)
