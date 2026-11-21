"""Figures and portable report regenerated entirely from saved results/predictions."""
import csv
import html
import json
import zipfile
from pathlib import Path
import numpy as np
from common import read_json,save,LEVELS_MS,STAGES
import reporting as legacy
import matplotlib.pyplot as plt


def export(run,name,fig):
    fig.text(.5,.005,'Previously inspected test set: exploratory reference comparison',ha='center',fontsize=8,color='#744')
    legacy_export(run,name,fig)

legacy_export=legacy.export
legacy.export=export


def histories(run,name,history):
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for prefix in ('train','validation'):
        axes[0].plot([r['step'] for r in history],[r[prefix+'_log_loss'] for r in history],label=prefix)
        rows=[r for r in history if prefix+'_macro_f1' in r]
        if rows:axes[1].plot([r['step'] for r in rows],[r[prefix+'_macro_f1'] for r in rows],label=prefix)
    for ax in axes:ax.set_xlabel(history[0]['axis']);ax.legend()
    axes[0].set_ylabel('Unweighted cross entropy');axes[1].set(ylabel='Five-stage macro-F1',ylim=(0,1))
    scope=history[-1].get('monitoring_scope','See protocol')
    fig.suptitle(name+'\n'+scope,fontsize=10);export(run,'learning_'+name,fig)


def build_report(run):
    val=read_json(run/'validation_results.json');sel=read_json(run/'selection.json');winner=sel['winner']
    reports=read_json(run/'test_results.json');test=reports['five_stage'];prep=read_json(run/'preparation_manifest.json')
    names=sorted(val,key=lambda n:-val[n]['selection_score'])
    for path in sorted(run.glob('training_history_*.json')):histories(run,path.stem.removeprefix('training_history_'),read_json(path))
    fig,ax=plt.subplots(figsize=(11,6));ax.barh(names[::-1],[val[n]['selection_score'] for n in names[::-1]],color=['#e78b28' if n==winner else '#4477aa' for n in names[::-1]])
    ax.set(xlabel='Mean macro-F1, all common validation epochs at four noise levels',xlim=(0,1),title='Full validation comparison');export(run,'validation_comparison',fig)
    bounds={str(n):legacy.bootstrap_interval(test[str(n)]) for n in LEVELS_MS};save(run/'test_uncertainty.json',{'method':'1000 participant bootstrap replicates','macro_f1_95_intervals':bounds})
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for name in names:axes[0].plot(LEVELS_MS,[val[name]['validation'][str(n)]['macro_f1'] for n in LEVELS_MS],label=name)
    axes[0].legend(fontsize=6);axes[0].set(title='Validation comparison',ylabel='Macro-F1',xlabel='Target interval MAE (ms)',ylim=(0,1))
    axes[1].plot(LEVELS_MS,[test[str(n)]['macro_f1'] for n in LEVELS_MS],'o-',label='Reference test')
    axes[1].vlines(LEVELS_MS,[bounds[str(n)][0] for n in LEVELS_MS],[bounds[str(n)][1] for n in LEVELS_MS],label='95% participant interval')
    axes[1].plot(LEVELS_MS,[val[winner]['validation'][str(n)]['macro_f1'] for n in LEVELS_MS],'s--',label='Validation')
    axes[1].set(title=winner,xlabel='Target interval MAE (ms)',ylabel='Macro-F1',ylim=(0,1));axes[1].legend(fontsize=8);export(run,'noise_robustness',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,part,rr in [(axes[0],'Validation',val[winner]['validation']),(axes[1],'Reference test',test)]:
        for n in LEVELS_MS:
            legacy.confusion(run,part.lower().replace(' ','_'),n,rr[str(n)])
            ax.plot(STAGES,[rr[str(n)]['per_stage'][s]['f1'] for s in STAGES],label=f'{n} ms')
        ax.set(title=part,ylabel='Stage F1',ylim=(0,1));ax.legend()
    export(run,'stage_f1',fig)
    fig,ax=plt.subplots(figsize=(8,4));ax.boxplot([[r['macro_f1'] for r in test[str(n)]['per_participant'].values()] for n in LEVELS_MS])
    ax.set(xticks=range(1,5),xticklabels=[f'{n} ms' for n in LEVELS_MS],ylabel='Participant macro-F1',ylim=(0,1));export(run,'participant_variation',fig)
    importance=read_json(run/'feature_importance_validation.json');method=read_json(run/'feature_importance_method.json')['method']
    top=importance[:25][::-1];fig,ax=plt.subplots(figsize=(10,max(4,len(top)*.25)))
    ax.barh([r['feature'] for r in top],[r['mean_macro_f1_drop'] for r in top],xerr=[r['std'] for r in top]);ax.axvline(0,color='gray',lw=.5)
    ax.set(xlabel='Validation macro-F1 decrease',title=method,);export(run,'feature_importance_validation',fig)
    coverage=read_json(run/'coverage.json');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k,p in enumerate(('train','validation','test')):
        counts=prep['counts'][p]
        for i,key in enumerate(('eligible','common','expanded')):axes[0].bar(k+(i-1)*.25,counts[key],.25,label=key if k==0 else None)
    axes[0].set(xticks=range(3),xticklabels=['Train','Validation','Test'],ylabel='Epoch count',title='Quality coverage, no fitting caps');axes[0].legend()
    means=[np.mean([r['realized_mae_ms'][str(n)] for r in coverage]) for n in LEVELS_MS]
    stds=[np.std([r['realized_mae_ms'][str(n)] for r in coverage]) for n in LEVELS_MS]
    axes[1].errorbar(LEVELS_MS,means,yerr=stds,fmt='o');axes[1].plot(LEVELS_MS,LEVELS_MS,'k:');axes[1].set(xlabel='Target MAE (ms)',ylabel='Realized MAE (ms)',title='Timing jitter audit');export(run,'coverage_jitter',fig)
    diagnostics={}
    for n in LEVELS_MS:
        with (run/f'test_predictions_{n}ms.csv').open() as f:rows=list(csv.DictReader(f))
        diagnostics[str(n)]=legacy.probability_plots(run,n,rows)
        for pid in sorted({r['participant'] for r in rows})[:3]:
            record=sorted([r for r in rows if r['participant']==pid],key=lambda r:int(r['epoch']))
            epochs=np.array([int(r['epoch']) for r in record]);timeline=np.arange(epochs.min(),epochs.max()+1)
            fig,axes=plt.subplots(2,1,sharex=True,figsize=(11,4))
            for ax,key in zip(axes,['reference','prediction']):
                yy=np.full(len(timeline),np.nan);yy[epochs-timeline[0]]=[int(r[key]) for r in record]
                ax.step(timeline/120,yy,where='post');ax.set(yticks=range(5),yticklabels=STAGES,ylim=(4.5,-.5),ylabel=key)
            axes[1].set_xlabel('Hours from recording start; gaps = excluded epochs');fig.suptitle(f'Participant {pid}, {n} ms');export(run,f'hypnogram_{pid}_{n}ms',fig)
    save(run/'probability_diagnostics.json',diagnostics)
    with (run/'validation_comparison.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['candidate','mean_validation_macro_f1','training_targets_per_pass'])
        w.writerows((n,val[n]['selection_score'],val[n]['training_epochs']) for n in names)
    with (run/'reference_test_summary.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['noise_ms','epochs','accuracy','macro_f1','kappa']+[s+'_f1' for s in STAGES])
        for n in LEVELS_MS:
            r=test[str(n)];w.writerow([n,r['n_epochs'],r['accuracy'],r['macro_f1'],r['kappa']]+[r['per_stage'][s]['f1'] for s in STAGES])
    lines=['# MESA full-data improvement comparison','',f'Selected model: **{winner}**.','',
           '**Previously inspected test set: exploratory reference comparison. No claim of independent confirmation or real BCG validation.**','',
           'Selection uses every original-common validation epoch at 0/20/40/60 ms, with participant splits preserved. No training or validation epoch caps.', '',
           '## Reference test results','','| MAE | Accuracy | Macro-F1 | Kappa |','|---|---:|---:|---:|']
    for n in LEVELS_MS:
        r=test[str(n)];k='undefined' if r['kappa'] is None else f"{r['kappa']:.4f}"
        lines.append(f"| {n} ms | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {k} |")
    lines+=['','## Coverage','',json.dumps(prep['counts'],indent=2),'',
            '## Interpretation','',
            'Forest tree counts and boosting rounds describe model growth; temporal curves use neural training epochs. These are iteration curves, not training-set-size learning curves. Tree training diagnostics use up to 5,000 fixed training epochs and all clean validation epochs. TCN curves use all training targets in a mixed-noise pass and all common validation epochs averaged across four noise levels. Early stopping selects the TCN checkpoint from validation only. Unweighted diagnostic losses differ from class-weighted fitting objectives.','',
            'All annotation-stage columns are ignored. Inputs are beat timestamps and normal-beat flags. The 140 HRV inputs are the original 54, sixteen short-window measures, and seventy per-night relative copies. TCNs additionally receive missingness indicators and, for dual variants, 2-Hz RR/relative RR/validity streams. Per-night normalization and centered sequence windows require offline whole-night use.','',
            'Validation importance: '+method+'. Negative changes can arise from noise or feature redundancy. Bootstrap uncertainty resamples participants. Undefined participant kappa is saved as null. See source/bcg_full_training/PROTOCOL.md and source/bcg_jitter_staging/FEATURES.md for complete methods.','',
            'The prior capped RF control is evaluated only on full validation. If it wins, its saved reference-test predictions are reused. The newly expanded cohort has a different target population and is reported separately.','',
            '## Validation comparison','','| Candidate | Mean macro-F1 |','|---|---:|']
    lines += [f"| {n} | {val[n]['selection_score']:.4f} |" for n in names]
    if reports['expanded']:
        lines+=['','## Expanded target reference results','','| MAE | Epochs | Accuracy | Macro-F1 |','|---|---:|---:|---:|']
        for n in LEVELS_MS:
            r=reports['expanded'][str(n)];lines.append(f"| {n} | {r['n_epochs']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} |")
    lines+=['','## Figures','']
    for p in sorted((run/'figures').glob('*.png')):lines += [f'### {p.stem.replace("_"," ")}',f'![{p.stem}](figures/{p.name})','']
    (run/'REPORT.md').write_text('\n'.join(lines))
    body=''.join(f'<h2>{html.escape(p.stem)}</h2><img loading="lazy" src="figures/{p.name}">' for p in sorted((run/'figures').glob('*.png')))
    (run/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>MESA full-data results</title><style>body{max-width:1200px;margin:32px auto;font:16px system-ui}img{width:100%}</style><h1>MESA full-data comparison</h1><p>Selected: '+html.escape(winner)+'. Previously inspected test set: exploratory comparison. Read REPORT.md for tables and methods.</p>'+body)
    with zipfile.ZipFile(run/'results_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.csv','.md','.html','.png','.pdf','.py','.slurm','.txt'} and not p.name.endswith('.incomplete.csv'):
                z.write(p,p.relative_to(run))
