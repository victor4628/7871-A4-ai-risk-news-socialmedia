"""Retrieve public sources; frozen scope: 8-30 September 2026, New York time."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/python_deps'))
import json, time, re, html, csv, hashlib, urllib.request, urllib.parse
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
NY=ZoneInfo('America/New_York')
RAW=ROOT/'data/raw'; RAW.mkdir(parents=True,exist_ok=True)
OUT=ROOT/'data'; OUT.mkdir(exist_ok=True)
START=datetime(2026,9,8,tzinfo=NY); END=datetime(2026,10,1,tzinfo=NY)
GROUPS={'Semiconductors':['NVDA','AMD','AVGO','ARM','MU','TSM','ASML','MRVL'],
 'Cloud and platforms':['MSFT','GOOGL','AMZN','META','ORCL','IBM'],
 'Enterprise software':['PLTR','CRM','NOW','ADBE','SNOW'],
 'Servers':['DELL','HPE','SMCI'],'Autonomy':['TSLA']}
SYMBOLS=[s for g in GROUPS.values() for s in g]+['SPY','QQQ']
LOG=[]
def fetch(url,path,kind='json'):
 if path.exists(): return json.loads(path.read_text(encoding='utf-8')) if kind=='json' else path.read_text(encoding='utf-8')
 for attempt in range(3):
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (Academic AI sentiment study)'})
   with urllib.request.urlopen(req,timeout=30) as r: t=r.read().decode('utf-8')
   obj=json.loads(t) if kind=='json' else t
   path.parent.mkdir(parents=True,exist_ok=True);path.write_text(t,encoding='utf-8')
   LOG.append({'url':url,'file':str(path.relative_to(ROOT)),'retrieved_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(t.encode()).hexdigest()})
   return obj
  except Exception:
   if attempt==2:raise
   time.sleep(1+attempt)
def write_csv(name,rows):
 if not rows:return
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def prices(symbol):
 a=int(datetime(2026,9,4,tzinfo=timezone.utc).timestamp());b=int(END.timestamp())
 url=f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={a}&period2={b}&interval=1d&events=div%2Csplits'
 obj=fetch(url,RAW/'prices'/f'{symbol}.json')['chart']['result'][0]
 q=obj['indicators']['quote'][0];adj=obj['indicators']['adjclose'][0]['adjclose'];rows=[]
 for i,t in enumerate(obj['timestamp']):
  day=datetime.fromtimestamp(t,NY).date().isoformat()
  if day>'2026-09-30':continue
  rows.append({'date':day,'symbol':symbol,'open':q['open'][i],'close':q['close'][i],'adj_close':adj[i],'high':q['high'][i],'low':q['low'][i],'volume':q['volume'][i],'source_url':url})
 return rows
def news_day(day):
 date=day.date().isoformat();url=f'https://www.dataaihub.co/daily/{date}'
 soup=BeautifulSoup(fetch(url,RAW/'news'/f'{date}.html','text'),'html.parser')
 rows=[]
 for article in soup.find_all('article'):
  h=article.find('h2');link=next((a for a in article.find_all('a') if 'Read original article' in a.get_text(' ',strip=True)),None)
  if not h or not link:continue
  title=h.get_text(' ',strip=True);u=link.get('href','');domain=urllib.parse.urlparse(u).hostname or ''
  company=article.find('a',href=re.compile('^/companies/'))
  rows.append({'id':hashlib.sha256(u.encode()).hexdigest()[:16],'date':date,'available_at':datetime.combine(day.date(),datetime.max.time(),NY).isoformat(),
   'title':title,'url':u,'domain':domain.lower().removeprefix('www.'),'company_tag':company.get_text(' ',strip=True) if company else '',
   'archive_url':url,'timestamp_quality':'archive date only; assumed available at 23:59:59 NY'})
 return rows
def hn_window(a,b,key):
 params={'query':'AI','tags':'comment','restrictSearchableAttributes':'comment_text','numericFilters':f'created_at_i>={a},created_at_i<{b}','hitsPerPage':1000,'page':0}
 url='https://hn.algolia.com/api/v1/search_by_date?'+urllib.parse.urlencode(params)
 obj=fetch(url,RAW/'social'/f'{key}.json')
 if obj['nbHits']>1000:
  mid=(a+b)//2
  return hn_window(a,mid,key+'a')+hn_window(mid,b,key+'b')
 rows=[]
 for hit in obj['hits']:
  text=BeautifulSoup(hit.get('comment_text') or '', 'html.parser').get_text(' ',strip=True)
  if not re.search(r'\bAI\b',text,re.I):continue
  dt=datetime.fromtimestamp(hit['created_at_i'],NY)
  if not START<=dt<END:continue
  rows.append({'id':hit['objectID'],'date':dt.date().isoformat(),'created_at':dt.isoformat(),'author':hit.get('author'),'story_id':hit.get('story_id'),
   'story_title':hit.get('story_title'),'text':text,'url':'https://news.ycombinator.com/item?id='+hit['objectID'],'story_url':hit.get('story_url') or ''})
 return rows
def hn_day(day):return hn_window(int(day.timestamp()),int((day+timedelta(days=1)).timestamp()),day.date().isoformat())
def main():
 import argparse
 parser=argparse.ArgumentParser(description='Collect news and prices; X uses the separate budget-capped collector.')
 parser.add_argument('--include-hn',action='store_true',help='Optional legacy exploratory dataset, excluded from the X report.')
 args=parser.parse_args()
 days=[START+timedelta(days=i) for i in range((END-START).days)]
 streams=[('prices',SYMBOLS,prices,'prices.csv'),('news',days,news_day,'news_candidates.csv')]
 if args.include_hn:streams.append(('social',days,hn_day,'social_candidates.csv'))
 for label,items,fn,name in streams:
  allrows=[];errors=[]
  with ThreadPoolExecutor(max_workers=5) as pool:
   jobs={pool.submit(fn,item):str(item) for item in items}
   for job in as_completed(jobs):
    try:r=job.result();allrows.extend(r);print(label,jobs[job],len(r),flush=True)
    except Exception as e:errors.append({'item':jobs[job],'error':repr(e)});print('ERROR',label,jobs[job],repr(e),flush=True)
  write_csv(name,sorted(allrows,key=lambda r:(r.get('date',''),str(r.get('id',r.get('symbol',''))))))
  (OUT/f'{label}_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
 tweet=fetch('https://cdn.syndication.twimg.com/tweet-result?id=2097497037956891126&lang=en&token=1',RAW/'anchor_tweet.json')
 tweet['verified_ny_time']=datetime.fromisoformat(tweet['created_at'].replace('Z','+00:00')).astimezone(NY).isoformat()
 (OUT/'anchor_verification.json').write_text(json.dumps(tweet,ensure_ascii=False,indent=2),encoding='utf-8')
 manifest=OUT/'retrieval_manifest.json'
 previous=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else []
 # Preserve first retrieval metadata when subsequent runs reuse cached files.
 merged={r['file']:r for r in previous}
 for r in LOG:merged[r['file']]=r
 manifest.write_text(json.dumps(list(merged.values()),indent=2),encoding='utf-8')
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
