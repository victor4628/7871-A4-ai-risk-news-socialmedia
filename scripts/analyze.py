"""Frozen-window sentiment, basket returns, Granger tests and coverage comparisons."""
from pathlib import Path
import sys,json,re,warnings,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/python_deps'))
import numpy as np,pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import grangercausalitytests,adfuller
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collect_data import GROUPS,NY
warnings.filterwarnings('ignore',category=FutureWarning)
DATA=ROOT/'data';OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
US={'bloomberg.com','forbes.com','nytimes.com','washingtonpost.com','wsj.com','cnbc.com','cnn.com','foxnews.com','nbcnews.com','cbsnews.com','abcnews.com','apnews.com','politico.com','latimes.com','usatoday.com','usnews.com','fortune.com','marketwatch.com','pbs.org','npr.org','newsweek.com','axios.com'}
INTL={'bbc.com','bbc.co.uk','theguardian.com','reuters.com','dw.com','aljazeera.com','theglobeandmail.com','ft.com','lemonde.fr','economist.com'}
TECH={'techcrunch.com','the-decoder.com','thenewstack.io','officechai.com','arstechnica.com','wired.com','siliconangle.com','gizmodo.com','pymnts.com','spectrum.ieee.org','marktechpost.com','technologyreview.com','theverge.com','venturebeat.com','tomshardware.com','zdnet.com','techradar.com','fastcompany.com','theregister.com','engadget.com','businessinsider.com'}
CORP={'openai.com','anthropic.com','cloud.google.com','blog.google','deepmind.google','ai.google','aws.amazon.com','aboutamazon.com','developer.nvidia.com','nvidianews.nvidia.com','blogs.nvidia.com','nvidia.com','databricks.com','huggingface.co','microsoft.com','blogs.microsoft.com','news.microsoft.com','oracle.com','meta.com','about.fb.com','engineering.fb.com','ai.meta.com','research.google','ai.google.dev'}
SOCIAL={'facebook.com','linkedin.com','x.com','twitter.com','reddit.com','youtube.com'}
RISK=re.compile(r'\b(risks?|risky|danger\w*|threat\w*|extinct\w*|doom\w*|kill\w*|catastroph\w*|misalign\w*|misuse|harm\w*|unsafe|safety|safe|fraud|scam\w*|cyberattack\w*|hack\w*|suicid\w*|psychosis|deepfake\w*|bias|biased|privacy|gambling|reckless|irresponsible|uncontroll\w*|loss of control)\b',re.I)
EXIST=re.compile(r'extinct|kill (?:us|all|human)|doom|catastroph|superintellig|misalign|gambling with|loss of control|pace the frontier',re.I)
BENEFIT=re.compile(r'\b(benefit\w*|improv\w*|boost\w*|productiv\w*|breakthrough\w*|success\w*|help\w*|advanc\w*|accelerat\w*|efficien\w*|save\w*|saving\w*|protect\w*)\b',re.I)
def category(domain):
 if domain in US:return 'US major'
 if domain in INTL:return 'International major'
 if domain in TECH:return 'Technology media'
 if domain in CORP or domain.endswith('.google') or domain.endswith('.nvidia.com'):return 'Corporate and research'
 if domain in SOCIAL:return 'Social link'
 return 'Other outlets'
def topic(text):
 if re.search(r'Coxon|Hubinger|researcher.{0,25}(quit|resign)|gambling with our lives',text,re.I):return 'Insider warning'
 if re.search(r'safety|safe|risk|extinct|doom|superintelligen|regulat|court|sued|lawsuit|hack|privacy|FTC|White House|Trump',text,re.I):return 'Risk and governance'
 if re.search(r'invest|fund|billion|million|IPO|valuat|bond|deal|spend|revenue|earnings|profit|stock|shares',text,re.I):return 'Finance and investment'
 return 'Technology and use'
def clean_headline(t):
 parts=re.split(r'\s+-\s+',t)
 if len(parts)>1 and (len(parts[-1].split())<=5):
  suffix=parts[-1].lower()
  if re.search(r'\.com|\.org|\.io|news|bloomberg|guardian|reuters|forbes|cnbc|cnn|pbs|wral|fortune|politico|open ai|google|techcrunch|yahoo|times|abc|fox|press',suffix):t=' - '.join(parts[:-1])
 return t
vader=SentimentIntensityAnalyzer()
negfile=DATA/'raw/opinion_lexicon/negative-words.txt'
neg=set(l.strip() for l in negfile.read_text(encoding='latin1').splitlines() if l and not l.startswith(';'))
def score(df,col):
 df['compound']=df[col].map(lambda t:vader.polarity_scores(t)['compound'])
 df['tone']=np.select([df.compound>=.05,df.compound<=-.05],['positive','negative'],default='neutral')
 df['negative']=(df.tone=='negative').astype(int)
 df['positive']=(df.tone=='positive').astype(int)
 df['words']=df[col].map(lambda t:len(re.findall(r"[a-zA-Z]+(?:'[a-z]+)?",t)))
 df['negative_word_share']=df[col].map(lambda t:sum(w.lower() in neg for w in re.findall(r"[a-zA-Z]+(?:'[a-z]+)?",t))/max(1,len(re.findall(r"[a-zA-Z]+(?:'[a-z]+)?",t))))
 df['risk_theme']=df[col].str.contains(RISK).astype(int)
 df['existential_theme']=df[col].str.contains(EXIST).astype(int)
 df['benefit_theme']=df[col].str.contains(BENEFIT).astype(int)
 return df
def summarize(df,key):
 return df.groupby(key).agg(n=('compound','size'),mean=('compound','mean'),negative_share=('negative','mean'),positive_share=('positive','mean'),risk_share=('risk_theme','mean'),existential_share=('existential_theme','mean'),negative_words=('negative_word_share','mean'))
def bin_to_close(times,closes):
 a=pd.DatetimeIndex(pd.to_datetime(times,utc=True)).asi8;b=closes.asi8
 idx=np.searchsorted(b,a,side='left')
 return [closes[i].date().isoformat() if i<len(closes) else None for i in idx]
def test_pair(df,y,x,lag,label,primary=False):
 d=df[[y,x]].dropna()
 res=grangercausalitytests(d.to_numpy(),maxlag=[lag])[lag]
 f,p,dfden,dfnum=res[0]['ssr_ftest'];model=res[1][1]
 beta=model.params[lag:2*lag];ci=model.conf_int()[lag:2*lag]
 lb=float(acorr_ljungbox(model.resid,lags=[1],return_df=True).lb_pvalue.iloc[0])
 return {'specification':label,'outcome':y,'predictor':x,'lags':lag,'aligned_rows':len(d),'n':int(model.nobs),'F':float(f),'df_num':int(dfnum),'df_den':int(dfden),'p':float(p),'beta':float(beta[0]),'ci_low':float(ci[0,0]),'ci_high':float(ci[0,1]),'beta_sum':float(beta.sum()),'lb_p':lb,'primary':primary}
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--social-source',choices=['x','hn'],default='x')
 parser.add_argument('--min-followers',type=int,default=100)
 parser.add_argument('--min-views',type=int,default=100)
 args=parser.parse_args()
 news=pd.read_csv(DATA/'news_candidates.csv').sort_values('date')
 original_news=len(news)
 news['text']=news.title.map(clean_headline)
 news['normalized_title']=news.text.str.lower().str.replace(r'\W+',' ',regex=True).str.strip()
 news=news.drop_duplicates('url').drop_duplicates('normalized_title').copy()
 news['category']=news.domain.map(category);news['topic']=news.text.map(topic)
 news=score(news,'text')
 editorial=news[~news.category.isin(['Corporate and research','Social link'])].copy()
 if args.social_source=='x':
  manifest=json.loads((DATA/'x/stratified_manifest.json').read_text())
  if not manifest['completed']:raise ValueError('X collection is incomplete; do not fit full-period models from a partial sample.')
  from filter_x import read_posts,apply_filter,clean_for_analysis
  social,_=read_posts()
  social,exposure_audit=apply_filter(social,args.min_followers,args.min_views)
  (OUT/'x_exposure_filter.json').write_text(json.dumps(exposure_audit,indent=2),encoding='utf-8')
  raw_x_count=len(social)
  unfiltered,cleaning_audit=clean_for_analysis(social)
  (OUT/'x_cleaning_audit.json').write_text(json.dumps(cleaning_audit,indent=2),encoding='utf-8')
  unfiltered=score(unfiltered,'text')
  unfiltered['story_id']=unfiltered.conversation_id
  unfiltered.to_csv(OUT/'social_unfiltered_scored.csv',index=False,encoding='utf-8-sig')
  # Apply the exposure gate to the same cleaned universe, so comparisons are
  # exact subsets rather than different choices among repeated author/text posts.
  social=unfiltered[unfiltered.exposure_eligible].copy()
  social['story_id']=social.conversation_id
 else:social=pd.read_csv(DATA/'social_candidates.csv',dtype={'id':str}).drop_duplicates('id').copy()
 original_social=len(social)
 if args.social_source=='x':original_social=raw_x_count
 social['normalized']=social.text.str.lower().str.replace(r'\s+',' ',regex=True).str.strip()
 social=social[social.text.str.split().str.len()>=5]
 social=social.drop_duplicates(['author','normalized']).copy()
 if args.social_source=='x':
  daily=social.groupby('date').size()
  if len(daily)!=23 or daily.min()<20:
   raise ValueError('Filtered X sample lacks adequate coverage of all 23 dates; do not silently fit incomplete-period models.')
 social=score(social,'text')
 for label,df in [('news_all',news),('news_editorial',editorial),('social',social)]:
  df.to_csv(OUT/f'{label}_scored.csv',index=False,encoding='utf-8-sig')
  summarize(df,'date').to_csv(OUT/f'{label}_daily.csv')
  df['phase']=np.select([df.date<='2026-09-14',df.date<='2026-09-22'],['Sep 8-14','Sep 15-22'],default='Sep 23-30')
  summarize(df,'phase').to_csv(OUT/f'{label}_phases.csv')
 p=pd.read_csv(DATA/'prices.csv');wide=p.pivot(index='date',columns='symbol',values='adj_close').sort_index()
 assert wide.notna().all().all() and len(wide)==18 and len(wide.columns)==25
 stocks=[s for a in GROUPS.values() for s in a]
 wealth=wide[stocks].div(wide[stocks].iloc[0]).mean(axis=1)
 returns=pd.DataFrame(index=wide.index)
 returns['basket_wealth']=wealth;returns['basket_return']=np.log(wealth).diff()*100
 for benchmark in ['SPY','QQQ']:
  returns[benchmark+'_wealth']=wide[benchmark]/wide[benchmark].iloc[0]
  returns[benchmark+'_return']=np.log(wide[benchmark]).diff()*100
  returns['excess_'+benchmark]=returns.basket_return-returns[benchmark+'_return']
 returns['basket_simple_return']=wealth.pct_change()*100
 for group,symbols in GROUPS.items():returns[group+'_wealth']=wide[symbols].div(wide[symbols].iloc[0]).mean(axis=1)
 returns.to_csv(OUT/'returns.csv')
 constituents=[]
 for g,syms in GROUPS.items():
  for s in syms:constituents.append({'symbol':s,'group':g,'start_adj_close':wide[s].iloc[0],'end_adj_close':wide[s].iloc[-1],'return_pct':100*(wide[s].iloc[-1]/wide[s].iloc[0]-1),'contribution_pp':100*(wide[s].iloc[-1]/wide[s].iloc[0]-1)/23})
 pd.DataFrame(constituents).to_csv(OUT/'constituent_returns.csv',index=False)
 closes=pd.DatetimeIndex([pd.Timestamp(d+' 16:00',tz=NY) for d in wide.index if d>='2026-09-08'])
 editorial['close_bin']=bin_to_close(editorial.available_at,closes)
 social['close_bin']=bin_to_close(social.created_at,closes)
 aligned=returns.loc['2026-09-09':].copy() # Sep 8 is left-truncated for both channels
 for label,df in [('news',editorial),('social',social)]:
  grouped=summarize(df.dropna(subset=['close_bin']),'close_bin')
  aligned[label+'_sentiment']=grouped['mean'];aligned[label+'_n']=grouped.n
  aligned['delta_'+label]=aligned[label+'_sentiment'].diff()
  aligned[label+'_negativity']=grouped.negative_share
  aligned[label+'_risk']=grouped.risk_share
  if label=='social':
   aligned['social_thread_balanced']=df.groupby(['close_bin','story_id']).compound.mean().groupby(level=0).mean()
   aligned['social_author_balanced']=df.groupby(['close_bin','author']).compound.mean().groupby(level=0).mean()
   aligned['delta_social_thread_balanced']=aligned.social_thread_balanced.diff()
   aligned['delta_social_author_balanced']=aligned.social_author_balanced.diff()
 aligned.to_csv(OUT/'model_data.csv')
 tests=[]
 for ch in ['news','social']:
  x='delta_'+ch
  tests.append(test_pair(aligned,'basket_return',x,1,ch+' change -> basket',True))
  tests.append(test_pair(aligned,x,'basket_return',1,'basket -> '+ch+' change',True))
  for y in ['basket_return','excess_SPY','excess_QQQ']:
   for lag in [1,2]:
    if y=='basket_return' and lag==1:continue
    tests.append(test_pair(aligned,y,x,lag,f'{ch} change -> {y}, lag {lag}'))
  tests.append(test_pair(aligned,'basket_return',ch+'_sentiment',1,ch+' level -> basket'))
 for x in ['delta_social_thread_balanced','delta_social_author_balanced']:
  tests.append(test_pair(aligned,'basket_return',x,1,x+' -> basket'))
 t=pd.DataFrame(tests);t['holm_p']=np.nan
 t.loc[t.primary,'holm_p']=multipletests(t.loc[t.primary,'p'],method='holm')[1]
 t.to_csv(OUT/'granger_results.csv',index=False)
 diagnostics=[]
 for c in ['basket_return','delta_news','delta_social']:
  a=adfuller(aligned[c].dropna(),maxlag=0,autolag=None,regression='c')
  diagnostics.append({'series':c,'ADF_stat':a[0],'p':a[1],'n':a[3]})
 pd.DataFrame(diagnostics).to_csv(OUT/'stationarity.csv',index=False)
 # COVID-style comparisons use headline tone, with source-domain clustered uncertainty.
 news['neg_z']=(news.negative_word_share-news.negative_word_share.mean())/news.negative_word_share.std(ddof=0)
 bias=news[news.category!='Social link'].copy()
 bias['log_words']=np.log1p(bias.words)
 model=smf.ols('negative ~ C(category, Treatment(reference="International major")) + C(date) + C(topic) + log_words + I(log_words**2)',bias).fit(cov_type='cluster',cov_kwds={'groups':bias.domain})
 model2=smf.ols('neg_z ~ C(category, Treatment(reference="International major")) + C(date) + C(topic) + log_words + I(log_words**2)',bias).fit(cov_type='cluster',cov_kwds={'groups':bias.domain})
 reg=[]
 for label,m in [('Negative probability',model),('Hu Liu z score',model2)]:
  for key in m.params.index:
   if 'category' in key:reg.append({'outcome':label,'term':key,'coefficient':m.params[key],'se':m.bse[key],'p':m.pvalues[key],'low':m.conf_int().loc[key,0],'high':m.conf_int().loc[key,1],'n':int(m.nobs),'domains':bias.domain.nunique()})
 pd.DataFrame(reg).to_csv(OUT/'bias_regressions.csv',index=False)
 summarize(news,'category').to_csv(OUT/'source_comparison.csv')
 summarize(editorial,'topic').to_csv(OUT/'topic_comparison.csv')
 # Blind review sample: selected before exposing automated sentiment scores to reviewer.
 samples=[]
 for label,df,col in [('News',editorial,'text'),('Social',social,'text')]:
  sample=df.sample(n=20,random_state=7871)
  for _,r in sample.iterrows():samples.append({'channel':label,'id':r.id,'date':r.date,'text':r[col],'url':r.url})
 pd.DataFrame(samples).to_csv(OUT/'review_sample.csv',index=False,encoding='utf-8-sig')
 summary={'social_source':args.social_source,'raw_news':original_news,'deduplicated_news':len(news),'editorial_news':len(editorial),'corporate_news':int((news.category=='Corporate and research').sum()),'raw_social':original_social,'social':len(social),'social_authors':social.author.nunique(),'social_threads':social.story_id.nunique(),'news_domains':editorial.domain.nunique(),'dates':23,'price_days':17,
 'basket_return_pct':100*(wealth.iloc[-1]-1),'SPY_return_pct':100*(returns.SPY_wealth.iloc[-1]-1),'QQQ_return_pct':100*(returns.QQQ_wealth.iloc[-1]-1),'trough_date':wealth.idxmin(),'trough_return_pct':100*(wealth.min()-1),'positive_stocks':int(sum(r['return_pct']>0 for r in constituents)),
 'sectors':{g:100*(returns[g+'_wealth'].iloc[-1]-1) for g in GROUPS},'primary_tests':t[t.primary].to_dict('records'),'diagnostics':diagnostics}
 if args.social_source=='x':summary['exposure_filter']=exposure_audit
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:int(x)),encoding='utf-8')
 print(json.dumps(summary,indent=2));print('\nPHASES');print(pd.read_csv(OUT/'news_editorial_phases.csv').to_string(index=False));print(pd.read_csv(OUT/'social_phases.csv').to_string(index=False));print('\nSOURCE COMPARISON');print(pd.read_csv(OUT/'source_comparison.csv').to_string(index=False));print('\nBIAS');print(pd.DataFrame(reg).to_string(index=False))
if __name__=='__main__':main()
