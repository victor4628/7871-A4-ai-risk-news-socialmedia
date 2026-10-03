from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/python_deps'))
import numpy as np,pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller
from analyze import summarize,bin_to_close,test_pair,NY
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,matplotlib.dates as mdates
OUT=ROOT/'analysis';FIG=ROOT/'output/figures'
def main():
 n=pd.read_csv(OUT/'news_all_roberta.csv',dtype={'id':str});s=pd.read_csv(OUT/'social_roberta.csv',dtype={'id':str})
 for df in [n,s]:df['roberta_negative']=(df.roberta_tone=='negative').astype(int);df['roberta_positive']=(df.roberta_tone=='positive').astype(int)
 ed=n[~n.category.isin(['Corporate and research','Social link'])].copy()
 phases=[];daily={}
 for label,df in [('News',ed),('Social',s)]:
  df['phase']=np.select([df.date<='2026-09-14',df.date<='2026-09-22'],['Sep 8-14','Sep 15-22'],default='Sep 23-30')
  q=df.groupby('phase').agg(n=('id','size'),mean=('roberta_score','mean'),negative_share=('roberta_negative','mean'),positive_share=('roberta_positive','mean'))
  q['channel']=label;phases.append(q.reset_index())
  d=df.groupby('date').agg(n=('id','size'),mean=('roberta_score','mean'),negative_share=('roberta_negative','mean'),vader_negative=('negative','mean'),hu_liu_negative=('negative_word_share','mean'))
  d.to_csv(OUT/f'{label.lower()}_roberta_daily.csv');daily[label]=d
 pd.concat(phases).to_csv(OUT/'roberta_phases.csv',index=False)
 review=pd.read_csv(OUT/'qualitative_audit.csv',dtype={'id':str});audit=[]
 for label,df in [('News',n),('Social',s)]:
  d=review[review.channel==label].merge(df[['id','roberta_tone']],on='id')
  audit.append({'channel':label,'n':len(d),'vader_agreement':float((d.review_tone==d.automated_tone).mean()),'roberta_agreement':float((d.review_tone==d.roberta_tone).mean())})
 (OUT/'both_model_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 returns=pd.read_csv(OUT/'returns.csv',index_col='date')
 closes=pd.DatetimeIndex([pd.Timestamp(d+' 16:00',tz=NY) for d in returns.index if d>='2026-09-08'])
 aligned=returns.loc['2026-09-09':].copy()
 for label,df in [('news',ed),('social',s)]:
  df['close_bin']=bin_to_close(df.available_at if label=='news' else df.created_at,closes)
  aligned[label+'_roberta']=df.groupby('close_bin').roberta_score.mean()
  aligned['delta_'+label+'_roberta']=aligned[label+'_roberta'].diff()
 aligned.to_csv(OUT/'roberta_model_data.csv')
 tests=[]
 for ch in ['news','social']:
  x='delta_'+ch+'_roberta'
  tests.append(test_pair(aligned,'basket_return',x,1,'RoBERTa '+ch+' change -> basket',True))
  tests.append(test_pair(aligned,x,'basket_return',1,'RoBERTa basket -> '+ch+' change',True))
  tests.append(test_pair(aligned,'basket_return',x,2,'RoBERTa '+ch+' change -> basket, lag 2'))
  tests.append(test_pair(aligned,'excess_QQQ',x,1,'RoBERTa '+ch+' change -> excess QQQ'))
 t=pd.DataFrame(tests);t['holm_p']=np.nan;t.loc[t.primary,'holm_p']=multipletests(t.loc[t.primary,'p'],method='holm')[1]
 t.to_csv(OUT/'roberta_granger_results.csv',index=False)
 both=pd.concat([pd.read_csv(OUT/'granger_results.csv'),t],ignore_index=True)
 both['all_specs_holm_p']=multipletests(both.p,method='holm')[1];both['eight_primary_holm_p']=np.nan
 both.loc[both.primary,'eight_primary_holm_p']=multipletests(both.loc[both.primary,'p'],method='holm')[1]
 both.to_csv(OUT/'all_granger_results.csv',index=False)
 diag=[]
 for col in ['delta_news_roberta','delta_social_roberta']:
  a=adfuller(aligned[col].dropna(),maxlag=0,autolag=None);diag.append({'series':col,'ADF':a[0],'p':a[1],'n':a[3]})
 pd.DataFrame(diag).to_csv(OUT/'roberta_stationarity.csv',index=False)
 n['neg_z']=(n.negative_word_share-n.negative_word_share.mean())/n.negative_word_share.std(ddof=0)
 b=n[n.category!='Social link'].copy();b['log_words']=np.log1p(b.words);comparisons=[]
 for lab,y in [('VADER','negative'),('RoBERTa','roberta_negative'),('Hu Liu','neg_z')]:
  m=smf.ols(f'{y} ~ C(category, Treatment(reference="Corporate and research")) + C(date) + C(topic) + log_words + I(log_words**2)',b).fit(cov_type='cluster',cov_kwds={'groups':b.domain})
  for group in ['US major','Technology media','International major']:
   term=f'C(category, Treatment(reference="Corporate and research"))[T.{group}]'
   comparisons.append({'model':lab,'comparison':group+' vs corporate','coefficient':m.params[term],'low':m.conf_int().loc[term,0],'high':m.conf_int().loc[term,1],'p':m.pvalues[term],'n':int(m.nobs)})
  names=m.params.index;vec=np.zeros(len(names))
  vec[names.get_loc('C(category, Treatment(reference="Corporate and research"))[T.US major]')]=1
  vec[names.get_loc('C(category, Treatment(reference="Corporate and research"))[T.International major]')]=-1
  ct=m.t_test(vec);ci=ct.conf_int()[0]
  comparisons.append({'model':lab,'comparison':'US vs international major','coefficient':float(ct.effect[0]),'low':ci[0],'high':ci[1],'p':float(ct.pvalue),'n':int(m.nobs)})
 pd.DataFrame(comparisons).to_csv(OUT/'both_model_bias.csv',index=False)
 src=n.groupby('category').agg(n=('id','size'),vader_negative=('negative','mean'),roberta_negative=('roberta_negative','mean'),negative_words=('negative_word_share','mean'),risk_share=('risk_theme','mean'))
 src.to_csv(OUT/'both_model_sources.csv')
 # Final figure: separate panels expose model sensitivity rather than hide it.
 fig,axs=plt.subplots(2,1,figsize=(8.4,4.5),sharex=True)
 for ax,label,color in [(axs[0],'News','#254C45'),(axs[1],'Social','#66727B')]:
  d=daily[label].copy();d.index=pd.to_datetime(d.index)
  ax.plot(d.index,100*d.vader_negative.rolling(3,min_periods=1).mean(),label='VADER',color=color,linewidth=2)
  ax.plot(d.index,100*d.negative_share.rolling(3,min_periods=1).mean(),label='RoBERTa',color=color,linewidth=2,linestyle='--')
  social_label='X posts' if 'sample_window' in s.columns else 'HN comments'
  ax.set_ylabel(('News headlines' if label=='News' else social_label)+'\nNegative tone (%)');ax.set_ylim(0,75);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,loc='upper right',ncol=2,fontsize=9)
 axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=3));axs[1].xaxis.set_major_formatter(mdates.DateFormatter('Sep %d'));fig.tight_layout();fig.savefig(FIG/'sentiment_models.png',dpi=220,bbox_inches='tight');plt.close(fig)
 src=src.drop(index='Social link').reindex(['Corporate and research','Other outlets','Technology media','US major','International major']);fig,ax=plt.subplots(figsize=(8.4,2.8));y=np.arange(len(src))
 ax.barh(y-.18,100*src.vader_negative,height=.34,label='VADER',color='#254C45');ax.barh(y+.18,100*src.roberta_negative,height=.34,label='RoBERTa',color='#A0B9B0');ax.set_yticks(y,[f'{a} (n={int(b)})' for a,b in zip(src.index,src.n)]);ax.set_xlim(0,55);ax.set_xlabel('Headlines classified as negative (%)');ax.legend(frameon=False,ncol=2,loc='upper right',fontsize=9);ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(FIG/'source_models.png',dpi=220,bbox_inches='tight');plt.close(fig)
 print('PHASES',pd.concat(phases).to_string(index=False));print('AUDIT',audit);print('TESTS',t[['specification','p','beta','ci_low','ci_high','n']].to_string(index=False));print('BIAS',pd.DataFrame(comparisons).to_string(index=False));print('SOURCES',src.to_string());print('MIN CORRECTED',both.all_specs_holm_p.min())
if __name__=='__main__':main()
