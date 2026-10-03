from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/python_deps'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.stats.multitest import multipletests
from scipy.stats import linregress
OUT=ROOT/'analysis';FIG=ROOT/'output/figures';FIG.mkdir(parents=True,exist_ok=True)
GREEN='#254C45';GRAY='#66727B';RED='#B45B54';GOLD='#B39859'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#333333','axes.titleweight':'bold','savefig.dpi':220})
def save(fig,name):fig.savefig(FIG/name,bbox_inches='tight',facecolor='white');plt.close(fig)
def manual_audit():
 # AI-assisted blind reading of the fixed sample, before viewing automated scores.
 # Labels describe overall tone, not a human gold standard or AI-target stance.
 labels={'News':['neutral','neutral','negative','negative','negative','positive','negative','positive','positive','negative','neutral','positive','positive','positive','negative','neutral','neutral','neutral','positive','negative'],
 'Social':['neutral','neutral','negative','neutral','negative','negative','neutral','neutral','negative','negative','positive','negative','positive','neutral','negative','negative','neutral','negative','positive','negative']}
 reasons={'a71e4b786106b4a1':'Policy position; no explicit praise or fear.',
 '022a99ed331502b8':'Denial of extinction risk is reassuring; risk vocabulary can reverse classification.',
 'd746251ded6ddc31':'Negative tone toward doom advocates, not support for their AI-risk forecast.',
 '49829546':'Sarcastic praise precedes allegations of abuse; negative stance despite positive words.',
 '49682735':'Expresses hope about AI despite negative description of the current world.',
 '49650361':'Speculative takeover scenario despite a positive opening phrase.',
 '49692886':'Describes governance arrangements without explicit evaluative tone.'}
 d=pd.read_csv(OUT/'review_sample.csv',dtype={'id':str});rows=[]
 summary=json.loads((OUT/'summary.json').read_text())
 news_labels={str(row['id']):row for row in json.loads((OUT/'news_review_labels.json').read_text())}
 if set(news_labels)!=set(d.loc[d.channel=='News','id']):raise ValueError('News review labels do not match the fixed sample IDs; relabel without viewing scores.')
 x_labels={}
 if summary.get('social_source')=='x':
  label_file=OUT/'x_review_labels.json'
  if not label_file.exists():raise ValueError('Prepare blind labels for the new fixed X review sample before producing audit results.')
  x_labels={str(row['id']):row for row in json.loads(label_file.read_text())}
  if set(x_labels)!=set(d.loc[d.channel=='Social','id']):raise ValueError('X review labels do not match the current sample IDs.')
 for ch in ['News','Social']:
  sample=d[d.channel==ch]
  scored=pd.read_csv(OUT/('news_editorial_scored.csv' if ch=='News' else 'social_scored.csv'),dtype={'id':str}).set_index('id')
  for lab,(_,r) in zip(labels[ch],sample.iterrows()):
   if ch=='News':lab=news_labels[r.id]['review_tone']
   if ch=='Social' and x_labels:lab=x_labels[r.id]['review_tone']
   if lab not in ['negative','neutral','positive']:raise ValueError('Review label must be negative, neutral or positive.')
   rows.append({'channel':ch,'id':r.id,'date':r.date,'url':r.url,'review_tone':lab,'automated_tone':scored.loc[r.id,'tone'],'reason':reasons.get(r.id,'Overall tone assigned through blind qualitative reading.')})
   if ch=='Social' and x_labels:rows[-1]['reason']=x_labels[r.id].get('reason','AI-assisted blind review of X post tone.')
   if ch=='News':rows[-1]['reason']=news_labels[r.id]['reason']
 audit=pd.DataFrame(rows);audit.to_csv(OUT/'qualitative_audit.csv',index=False)
 metrics=[]
 for ch,g in audit.groupby('channel'):
  # Do not imply reviewer labels are independent human truth.
  acc=(g.review_tone==g.automated_tone).mean()
  classes=['negative','neutral','positive'];f1s=[]
  for c in classes:
   tp=((g.review_tone==c)&(g.automated_tone==c)).sum();fp=((g.review_tone!=c)&(g.automated_tone==c)).sum();fn=((g.review_tone==c)&(g.automated_tone!=c)).sum()
   f1s.append(2*tp/max(1,2*tp+fp+fn))
  metrics.append({'channel':ch,'n':len(g),'agreement':acc,'macro_f1_vs_review':float(np.mean(f1s))})
 (OUT/'audit_summary.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
 return metrics
def main():
 # sklearn is not needed; metrics are computed above to keep the environment minimal.
 a=manual_audit();print('AUDIT',a)
 tests=pd.read_csv(OUT/'granger_results.csv');tests['all_specs_holm_p']=multipletests(tests.p,method='holm')[1];tests.to_csv(OUT/'granger_results.csv',index=False)
 nd=pd.read_csv(OUT/'news_editorial_daily.csv',parse_dates=['date']);sd=pd.read_csv(OUT/'social_daily.csv',parse_dates=['date'])
 fig,axs=plt.subplots(2,1,figsize=(8.4,4.7),sharex=True)
 social_label='X posts' if json.loads((OUT/'summary.json').read_text()).get('social_source')=='x' else 'Hacker News comments'
 for d,label,col in [(nd,'News headlines',GREEN),(sd,social_label,GRAY)]:
  axs[0].plot(d.date,d['mean'],color=col,linewidth=1,alpha=.35)
  axs[0].plot(d.date,d['mean'].rolling(3,min_periods=1).mean(),color=col,linewidth=2.2,label=label)
  axs[1].plot(d.date,100*d.negative_share.rolling(3,min_periods=1).mean(),color=col,linewidth=2.2,label=label)
 axs[0].axhline(0,color='#BBBBBB',linewidth=.7);axs[0].set_ylabel('Mean tone score');axs[0].legend(loc='upper right',frameon=False,ncol=2,fontsize=9)
 axs[1].set_ylabel('Negative tone (%)');axs[1].set_ylim(0,65)
 for ax in axs:ax.grid(axis='y',alpha=.18)
 axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=3));axs[1].xaxis.set_major_formatter(mdates.DateFormatter('Sep %d'))
 fig.tight_layout();save(fig,'sentiment_trend.png')
 r=pd.read_csv(OUT/'returns.csv',parse_dates=['date']);fig,ax=plt.subplots(figsize=(8.4,3.6))
 for col,label,color in [('basket_wealth','23-stock AI basket',GREEN),('QQQ_wealth','QQQ',GOLD),('SPY_wealth','SPY',GRAY)]:ax.plot(r.date,100*(r[col]-1),label=label,color=color,linewidth=2.2)
 ax.axhline(0,color='#BBBBBB',linewidth=.7);ax.set_ylabel('Cumulative total return (%)');ax.legend(loc='upper left',frameon=False,ncol=3,fontsize=9);ax.grid(axis='y',alpha=.18)
 ax.xaxis.set_major_locator(mdates.DayLocator(interval=4));ax.xaxis.set_major_formatter(mdates.DateFormatter('Sep %d'));fig.tight_layout();save(fig,'basket_returns.png')
 c=pd.read_csv(OUT/'constituent_returns.csv').sort_values('return_pct');fig,ax=plt.subplots(figsize=(8.4,5.1))
 ax.barh(c.symbol,c.return_pct,color=[GREEN if x>=0 else RED for x in c.return_pct],height=.72)
 ax.axvline(0,color='#777777',linewidth=.8);ax.set_xlabel('Total return, Sep 4 close to Sep 30 close (%)');ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.15)
 for y,x in enumerate(c.return_pct):ax.text(x+(.4 if x>=0 else -.4),y,f'{x:+.1f}',va='center',ha='left' if x>=0 else 'right',fontsize=8)
 ax.set_xlim(-18,33);fig.tight_layout();save(fig,'constituent_returns.png')
 fig,axs=plt.subplots(1,2,figsize=(8.4,3.4),gridspec_kw={'wspace':.35})
 for ax,part in zip(axs,[c.iloc[:12],c.iloc[12:]]):
  ax.barh(part.symbol,part.return_pct,color=[GREEN if x>=0 else RED for x in part.return_pct],height=.7)
  ax.invert_yaxis();ax.axvline(0,color='#777777',linewidth=.7);ax.grid(axis='x',alpha=.15);ax.tick_params(axis='y',labelsize=8)
  for y,x in enumerate(part.return_pct):ax.text(x+(.5 if x>=0 else -.5),y,f'{x:+.1f}',va='center',ha='left' if x>=0 else 'right',fontsize=8)
  ax.set_xlim(-18,33);ax.set_xlabel('Total return (%)',fontsize=9)
 fig.tight_layout();save(fig,'constituent_returns_compact.png')
 src=pd.read_csv(OUT/'source_comparison.csv');src=src[src.category!='Social link'].sort_values('negative_share')
 fig,ax=plt.subplots(figsize=(8.4,2.7));ax.barh(src.category,src.negative_share*100,color=[GREEN if c!='Corporate and research' else GOLD for c in src.category])
 for y,(_,rr) in enumerate(src.iterrows()):ax.text(rr.negative_share*100+.7,y,f'{rr.negative_share*100:.1f}%  (n={rr.n})',va='center',fontsize=9)
 ax.set_xlim(0,57);ax.set_xlabel('Headlines classified as negative (%)');ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,'source_negativity.png')
 metrics={}
 for label,d in [('news',nd),('social',sd)]:
  trend=linregress(np.arange(len(d)),d.negative_share)
  metrics[label]={'negative_share_slope_pp_per_calendar_day':100*trend.slope,'unweighted_daily_trend_p':trend.pvalue}
 (OUT/'descriptive_trend_checks.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
 print('TREND',metrics);print('PRIMARY/SENSITIVITY HOLM MIN',tests.all_specs_holm_p.min())
 print('AUDIT MISMATCHES',pd.read_csv(OUT/'qualitative_audit.csv').query('review_tone != automated_tone').to_string(index=False))
if __name__=='__main__':main()
