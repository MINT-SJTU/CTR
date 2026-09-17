#!/usr/bin/env python3
"""Rebuild paper tables and descriptive summaries from the aggregate experiment export."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data'/'experiments.json'
D=json.loads(SOURCE.read_text(encoding='utf-8'))
FRAMES={k:pd.DataFrame(v['rows']) for k,v in D.items() if isinstance(v,dict) and 'rows' in v}
a,b,c,e,f=[FRAMES[k] for k in ['e1_timing_bias','e2a_paired_freeze','e2b_heldout_schedule','e3_generality_constraints','e4_ablation']]
GROUPS=['L_to_R','R_to_L','Sync','Mix3','Ours']
MODELS=['DP','pi0.5','ACT']
LABEL={'L_to_R':r'$L\!\rightarrow\!R$','R_to_L':r'$R\!\rightarrow\!L$','Sync':'Sync','Mix3':'Mix-3','Ours':r'\method', 'pi0.5':r'$\pi_{0.5}$','B_orig':'Original','Fixed_Mask':'Fixed+Mask','Ours_no_IdleMask':r'\method{} w/o IdleMask','Ours_no_MainView':r'\method{} w/o main view','Real_Multi_optional':'Real-Multi','Fixed':'Fixed'}
for directory in ['tables','figures','data','notes']:(ROOT/directory).mkdir(exist_ok=True)
for k,df in FRAMES.items():df.to_csv(ROOT/'data'/f'{k}.csv',index=False)
def lab(x):return LABEL.get(x,str(x).replace('_',r'\_'))
def num(x,d=2):return '--' if pd.isna(x) else f'{x:.{d}f}'
def simple_table(name,cols,headers,rows):
 text='\\begin{tabular}{'+cols+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
 for r in rows:
  if r is None:text+='\\midrule\n'
  else:text+=' & '.join(str(x) for x in r)+r' \\'+'\n'
 text+='\\bottomrule\n\\end{tabular}\n';(ROOT/'tables'/f'{name}.tex').write_text(text)
def long_table(name,cols,headers,rows,caption,label):
 text=r'{\small'+'\n'+r'\begin{longtable}{'+cols+'}\n'+r'\caption{'+caption+'}'+r'\label{'+label+'}'+r'\\'+'\n\\toprule\n'
 h=' & '.join(headers)+r' \\'+'\n\\midrule\n'
 text+=h+r'\endfirsthead'+'\n\\toprule\n'+h+r'\endhead'+'\n'+r'\midrule\multicolumn{'+str(len(headers))+r'}{r}{Continued on next page}\\'+'\n'+r'\endfoot'+'\n\\bottomrule\n'+r'\endlastfoot'+'\n'
 for r in rows:
  text+='\\midrule\n' if r is None else ' & '.join(str(x) for x in r)+r' \\'+'\n'
 text+=r'\end{longtable}'+'\n}\n';(ROOT/'tables'/f'{name}.tex').write_text(text)
# E1: per model, three initial conditions and condition-matched diagnostics.
rows=[]
for model in MODELS:
 for group in GROUPS:
  s=a[(a.model==model)&(a.group==group)].set_index('condition');p=s.loc[['CL_left_complete','CR_right_complete']]
  w=(s.loc['CL_left_complete','wait_R_s']+s.loc['CR_right_complete','wait_L_s'])/2
  m=(s.loc['CL_left_complete','motion_L_cm']+s.loc['CR_right_complete','motion_R_cm'])/2
  rows.append([lab(model),lab(group),*[num(s.loc[x,'sr_pct'],1) for x in ['C0_both_unfinished','CL_left_complete','CR_right_complete']],num(w),num(m)])
 rows.append(None)
simple_table('e1_main','llrrrrr',['Model','Data',r'$\mathrm{SR}_0\uparrow$',r'$\mathrm{SR}_L\uparrow$',r'$\mathrm{SR}_R\uparrow$',r'$W_{\rm unfinished}\downarrow$',r'$M_{\rm complete}\downarrow$'],rows[:-1])
z=a[a.condition!='C0_both_unfinished'].copy();z['unfinished_wait']=np.where(z.condition=='CL_left_complete',z.wait_R_s,z.wait_L_s);z['completed_motion']=np.where(z.condition=='CL_left_complete',z.motion_L_cm,z.motion_R_cm)
e1pre=z.groupby('group')[['sr_pct','unfinished_wait','completed_motion']].mean()
# E2B summary, descriptive cell means (models x holdout conditions).
h=c.groupby('group').mean(numeric_only=True)
rows=[]
for g in GROUPS:
 r=h.loc[g];rows.append([lab(g),num(r.sr_seen_pct,1),num(r.sr_holdout_pct,1),num(r.sr_seen_pct-r.sr_holdout_pct,1),num(r.wait_L_holdout_s+r.wait_R_holdout_s),num(r.tcap_holdout_s)])
simple_table('e2b_main','lrrrrr',['Data',r'Seen SR $\uparrow$',r'Held-out SR $\uparrow$',r'Gap $\downarrow$',r'$W_L+W_R\downarrow$',r'$T_{\rm cap}\downarrow$'],rows)
# Generality: preserve task identities, success original->ours per model.
rows=[]
for task in ['S1','S2','S3','S4','R1','R2','R3','R4']:
 vals=[]
 for model in MODELS:
  s=e[(e.task==task)&(e.condition=='Default')&(e.model==model)].set_index('group');vals.extend([num(s.loc['B_orig','sr_pct'],1),num(s.loc['Ours','sr_pct'],1)])
 task_name={'S1':'S1: Bottles','S2':'S2: Drawer','S3':'S3: Scan object','S4':'S4: Dual shoes','R1':'R1: Block sorting','R2':'R2: Drawer','R3':'R3: Screw--sleeve','R4':'R4: Workpiece sorting'}[task]
 rows.append([task_name]+vals)
 if task=='S4':rows.append(None)
simple_table('e3_main','lrrrrrr',['Task','Orig.','CTR','Orig.','CTR','Orig.','CTR'],rows)
genpath=ROOT/'tables'/'e3_main.tex'
gentext=genpath.read_text().replace(r'\toprule',r'\toprule'+'\n'+r' & \multicolumn{2}{c}{DP} & \multicolumn{2}{c}{$\pi_{0.5}$} & \multicolumn{2}{c}{ACT} \\'+'\n'+r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}')
genpath.write_text(gentext)
# Constraint interventions; avoid arbitrary average seconds across unlike tasks.
rows=[]
for task in ['S2','S3','S4','R2','R3','R4']:
 cond='Occupy_release' if task[-1]=='4' else 'Prereq_delay_release';s=e[(e.task==task)&(e.condition==cond)].groupby('group').mean(numeric_only=True)
 for g in ['B_orig','Ours']:
  r=s.loc[g];viol=r.xr_pct if task[-1]=='4' else r.dvr_pct
  rows.append([task,lab(g),num(r.sr_pct,1),num(viol),num(r.release_latency_s),num(r.release_nonresponse_pct,1)])
 rows.append(None)
simple_table('constraints_main','llrrrr',['Task','Data',r'SR $\uparrow$',r'Violation $\downarrow$',r'Release (s) $\downarrow$',r'No response (\%) $\downarrow$'],rows[:-1])
# E4: real domain only; no DVR on independent task, no simulation Real-Multi mislabeled result.
ab=f[f.domain=='real'].groupby('group').mean(numeric_only=True)
ags=['Fixed','Fixed_Mask','Ours_no_IdleMask','Ours','Ours_no_MainView','Real_Multi_optional'];rows=[]
for g in ags:
 r=ab.loc[g];rows.append([lab(g),num(r.valid_action_elements/1e6,2),num(r.sr_pct,1),num(r.wait_s),num(r.motion_cm),num(r.delta_wait_s)])
simple_table('e4_main','lrrrrr',['Data','Labels (M)',r'SR $\uparrow$',r'$W\downarrow$',r'$M\downarrow$',r'$\Delta W\downarrow$'],rows)
# Full condition-level tables: imported intervals, not recomputed.
def ci(row,base,low,high,d=1):
 return num(row[base],d)+r'\, ['+num(row[low],d)+', '+num(row[high],d)+']'
rows=[]
cn={'C0_both_unfinished':'C0','CL_left_complete':'CL','CR_right_complete':'CR'}
for model in MODELS:
 for cond in cn:
  for g in GROUPS:
   r=a[(a.model==model)&(a.condition==cond)&(a.group==g)].iloc[0]
   rows.append([lab(model),cn[cond],lab(g),ci(r,'sr_pct','sr_ci_low','sr_ci_high'),num(r.tcap_s),num(r.wait_L_s)+' / '+num(r.wait_R_s),num(r.motion_L_cm)+' / '+num(r.motion_R_cm)])
  rows.append(None)
long_table('e1_full','lllrrrr',['Model','Init.','Data','SR [interval]',r'$T_{\rm cap}$',r'$W_L/W_R$',r'$M_L/M_R$'],rows[:-1],'All timing-distribution conditions. SR is in percent, times in seconds, and motion in centimeters. Bracketed intervals reproduce the condition-level interval fields.','tab:e1full')
rows=[]
for model in MODELS:
 for g in GROUPS:
  for arm in ['L','R']:
   for dur in [1.,2.,4.]:
    r=b[(b.model==model)&(b.group==g)&(b.frozen_arm==arm)&(b.freeze_duration_s==dur)].iloc[0]
    rows.append([lab(model),lab(g),arm,num(dur,0),ci(r,'delta_wait_s','delta_wait_ci_low','delta_wait_ci_high',2),num(r.delta_tcap_s),num(r.sr_control_pct,1)+' / '+num(r.sr_freeze_pct,1)])
  rows.append(None)
long_table('freeze_full','lllrrrr',['Model','Data','Arm','Pause',r'$\Delta W$ [interval]',r'$\Delta T_{\rm cap}$','SR: ctrl./pause'],rows[:-1],'All simulated paired-pause results. Each row records three training seeds, 60 matched pairs, 120 branch executions, and an eight-second response window.','tab:freezefull')
rows=[]
for model in MODELS:
 for cond in ['H1','H2_mirror','H3_other_phase']:
  for g in GROUPS:
   r=c[(c.model==model)&(c.condition==cond)&(c.group==g)].iloc[0]
   rows.append([lab(model),cond.split('_')[0],lab(g),num(r.sr_seen_pct,1),ci(r,'sr_holdout_pct','sr_holdout_ci_low','sr_holdout_ci_high'),num(r.tcap_holdout_s),num(r.wait_L_holdout_s)+' / '+num(r.wait_R_holdout_s)])
  rows.append(None)
long_table('heldout_full','lllrrrr',['Model','Split','Data','Seen SR','Held-out [interval]',r'$T_{\rm cap}$',r'$W_L/W_R$'],rows[:-1],'All held-out joint-progress results. Each row contains 90 seen and 90 held-out evaluations; H3 denotes the third late-mid/early-mid progress split.','tab:heldoutfull')
rows=[]
for task in ['S1','S2','S3','S4','R1','R2','R3','R4']:
 for cond in e[e.task==task].condition.unique():
  cc={'Default':'D','Prereq_delay_release':'P','Occupy_release':'X','Independent_after_cooperation':'I'}[cond]
  for model in MODELS:
   for g in ['B_orig','Ours']:
    r=e[(e.task==task)&(e.condition==cond)&(e.model==model)&(e.group==g)].iloc[0]
    rows.append([task,cc,lab(model),lab(g),ci(r,'sr_pct','sr_ci_low','sr_ci_high'),num(r.tcap_s),num(r.dvr_pct),num(r.xr_pct)])
  rows.append(None)
long_table('generality_full','llllrrrr',['Task','Cond.','Model','Data','SR [interval]',r'$T_{\rm cap}$','DVR','XR'],rows[:-1],'All task and condition results. D: default; P: prerequisite delay/release; X: workspace occupancy/release; I: independent-after-cooperation. Rates are in percent. A dash means not applicable.','tab:genfull')
rows=[]
for _,r in e.iterrows():
 cc={'Default':'D','Prereq_delay_release':'P','Occupy_release':'X','Independent_after_cooperation':'I'}[r.condition]
 rows.append([r.task,cc,lab(r.model),lab(r.group),num(r.cr_pct),num(r.sar_pct),num(r.dependency_exposure_pct,1),num(r.release_latency_s),num(r.release_nonresponse_pct,1)])
long_table('safety_full','llllrrrrr',['Task','Cond.','Model','Data','CR','SAR','Exposure','Release','No resp.'],rows,'Collision, safety intervention, dependency exposure, and release outcomes for every generality condition. All rates are percentages; release latency is in seconds. Raw interval columns are preserved in the accompanying CSV files.','tab:safetyfull')
rows=[]
for task in ['PutCub','R1','R3']:
 for model in ['DP','pi0.5']:
  for g in ags:
   r=f[(f.task==task)&(f.model==model)&(f.group==g)].iloc[0]
   # Do not present the sim-labeled Real-Multi entry as evidence from real demonstrations.
   if task=='PutCub' and g=='Real_Multi_optional':continue
   rows.append([task,lab(model),lab(g),ci(r,'sr_pct','sr_ci_low','sr_ci_high'),num(r.tcap_s),num(r.wait_s),num(r.motion_cm),num(r.delta_wait_s)])
  rows.append(None)
rows=[[('No mask' if v==r'\method{} w/o IdleMask' else 'No main' if v==r'\method{} w/o main view' else v) for v in row] if row is not None else None for row in rows]
long_table('ablation_full','lllrrrrr',['Task','Model','Data','SR [interval]',r'$T_{\rm cap}$',r'$W$',r'$M$',r'$\Delta W$'],rows[:-1],'Component-study results by task and model. The main-text summary uses the real R1/R3 conditions; the table retains the remaining reported ablations.','tab:ablfull')
# Figure inputs and headline claims; every number is derived from available aggregates.
claims={}
claims['e1_precomplete']=e1pre.to_dict(orient='index')
claims['freeze']=b.groupby(['group','freeze_duration_s'])[['delta_wait_s','delta_tcap_s','sr_control_pct','sr_freeze_pct']].mean().reset_index().to_dict(orient='records')
claims['heldout']=h[['sr_seen_pct','sr_holdout_pct','tcap_holdout_s','wait_L_holdout_s','wait_R_holdout_s']].to_dict(orient='index')
claims['e3_default']=e[e.condition=='Default'].groupby(['domain','group'])[['sr_pct','cr_pct','sar_pct']].mean().reset_index().to_dict(orient='records')
claims['e4_real']=ab.to_dict(orient='index')
claims['aggregation']='Unweighted descriptive macro-means over the explicitly selected model/condition/task cells; not pooled rollout estimates. No aggregate confidence intervals or p-values are constructed.'
(ROOT/'data'/'derived_summaries.json').write_text(json.dumps(claims,indent=2,allow_nan=True))
# Conditional arithmetic / provenance audit. These flags do not establish data fabrication.
audit={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'rows':{k:len(x) for k,x in FRAMES.items()},'conditional_count_grid_flags':{},'interval_widths':{}}
for key,df in FRAMES.items():
 if 'sr_pct' in df:rate,denom='sr_pct','n_episodes'
 elif 'sr_holdout_pct' in df:rate,denom='sr_holdout_pct','n_holdout_eval'
 else:rate,denom='sr_control_pct','n_pairs'
 flags=[]
 for i,row in df.iterrows():
  n=int(row[denom]);value=float(row[rate]);nearest=round(value*n/100)
  if abs(100*nearest/n-value)>.050000001:
   flags.append({'row_index':int(i),'model':row.model,'group':row.group,'task':row.task,'condition':row.condition,'rate':value,'denominator':n,'nearest_count':nearest,'nearest_rate':100*nearest/n})
 audit['conditional_count_grid_flags'][key]=flags
 lows=[x for x in df if x.endswith('_ci_low')]
 widths={}
 for low in lows:
  high=low.replace('_ci_low','_ci_high')
  if high in df:widths[low.replace('_ci_low','')]=sorted(set(round(float(x),8) for x in (df[high]-df[low]).dropna()))
 audit['interval_widths'][key]=widths
(ROOT/'notes'/'data_audit.json').write_text(json.dumps(audit,indent=2))
# Repeated assumptions are kept outside results, in a verification manifest.
print('Built tables from',sum(len(x) for x in FRAMES.values()),'aggregate rows.')
print('No input values, intervals, or missing values were replaced.')
