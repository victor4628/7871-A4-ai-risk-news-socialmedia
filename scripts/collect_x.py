"""Collect a fixed, stratified-time sample of X posts through TwitterAPI.io.

Dry run requires no key. Live collection requires an explicitly supplied cost cap.
Credentials are read from an environment variable or an ignored local key file.
API reference: https://docs.twitterapi.io/api-reference/endpoint/tweet_advanced_search
Pricing: https://twitterapi.io/pricing
"""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
from decimal import Decimal
import argparse,csv,hashlib,json,os,random,time,urllib.request,urllib.parse,urllib.error,shutil

ROOT=Path(__file__).resolve().parents[1]
NY=ZoneInfo('America/New_York')
START=datetime(2026,9,8,tzinfo=NY);END=datetime(2026,10,1,tzinfo=NY)
ENDPOINT='https://api.twitterapi.io/twitter/tweet/advanced_search'
QUERY='(AI OR "artificial intelligence" OR superintelligence) lang:en -filter:retweets'
RATE=Decimal('0.00015');PAGE_RESERVATION=RATE*20
SNOWFLAKE_EPOCH_MS=1288834974657

def id_bounds(window):
    # X IDs encode creation milliseconds in their high bits. max_id is inclusive.
    first=(window['start_epoch']*1000-SNOWFLAKE_EPOCH_MS)<<22
    last=((window['end_epoch']*1000-SNOWFLAKE_EPOCH_MS)<<22)-1
    return first,last

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return None  # Never forward the API key to a redirected destination.

def windows():
    """One uniformly selected five-minute interval within each four-hour stratum."""
    rng=random.Random(7871);result=[]
    for day in range(23):
        for stratum in range(6):
            base=START+timedelta(days=day,hours=stratum*4)
            begin=base+timedelta(minutes=rng.randrange(236))
            finish=begin+timedelta(minutes=5)
            result.append({'id':f'{base.date()}_s{stratum}',
                'start_ny':begin.isoformat(),'end_ny':finish.isoformat(),
                'start_epoch':int(begin.timestamp()),'end_epoch':int(finish.timestamp())})
    return result

def parse_time(value):
    try:return datetime.strptime(value,'%a %b %d %H:%M:%S %z %Y')
    except ValueError:return datetime.fromisoformat(value.replace('Z','+00:00'))

def backfill_cached_metadata(rows,log):
    """Recover metadata from saved responses without making paid requests."""
    for row in rows.values():
        row.setdefault('author_followers_at_retrieval','')
        row.setdefault('metrics_retrieved_utc','')
    for entry in log['requests']:
        path=ROOT/entry['file']
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=entry['sha256']:
            raise RuntimeError('Cached response hash mismatch during metadata backfill.')
        for post in json.loads(payload).get('tweets',[]):
            row=rows.get(str(post['id']))
            if row is None:continue
            row['author_followers_at_retrieval']=(post.get('author') or {}).get('followers','')
            row['metrics_retrieved_utc']=entry['retrieved_utc']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--pilot',action='store_true',help='One search to verify the initiating post.')
    parser.add_argument('--refetch',action='store_true',help='Explicitly start a new paid run after a completed run; preserve its ledger and local text.')
    parser.add_argument('--max-usd',type=Decimal,help='Explicit maximum estimated API charge for this run.')
    parser.add_argument('--key-file',type=Path,default=ROOT/'tmp/twitterapi_io.key')
    parser.add_argument('--max-pages-per-window',type=int,default=5)
    parser.add_argument('--request-interval',type=float,default=6.0,help='Minimum seconds between requests; free accounts require at least five.')
    args=parser.parse_args()
    if not 1<=args.max_pages_per_window<=5:parser.error('Page limit must be 1–5.')
    schedule=windows();base_query=QUERY
    if args.pilot:
        begin=datetime(2026,9,8,21,0,tzinfo=NY);finish=begin+timedelta(hours=1)
        schedule=[{'id':'anchor_pilot','start_ny':begin.isoformat(),'end_ny':finish.isoformat(),
            'start_epoch':int(begin.timestamp()),'end_epoch':int(finish.timestamp())}]
        base_query='from:EvanHub';args.max_pages_per_window=1
    plan={'period_start_ny':START.isoformat(),'period_end_exclusive_ny':END.isoformat(),
        'base_query':base_query,'query_type':'Latest','seed':7871,'windows':schedule,
        'sampling':('One anchor-post lookup, not a sentiment sample.' if args.pilot else
            'One random five-minute interval per four-hour stratum per day; capped pagination is recorded.'),
        'max_pages_per_window':args.max_pages_per_window,
        'boundary_method':'since_id and inclusive max_id derived from the fixed Unix-time window; returned timestamps validated client-side',
        'maximum_calls':len(schedule)*args.max_pages_per_window,
        'conservative_maximum_estimated_usd':str(len(schedule)*args.max_pages_per_window*PAGE_RESERVATION),
        'price_usd_per_returned_post':str(RATE),
        'limitations':'Indexed public English keyword-matching posts; not the full X population. Engagement counts are retrieval-time values.'}
    if args.dry_run:
        print(json.dumps({k:v for k,v in plan.items() if k!='windows'},indent=2))
        print('FIRST WINDOW',json.dumps(schedule[0]));print('LAST WINDOW',json.dumps(schedule[-1]));return
    if args.max_usd is None or args.max_usd<=0:parser.error('Live requests require a positive --max-usd cap.')
    key=os.environ.get('TWITTERAPI_IO_KEY','').strip()
    if not key and args.key_file.exists():key=args.key_file.read_text(encoding='utf-8').strip()
    if not key:parser.error('No API key: set TWITTERAPI_IO_KEY or save it in tmp/twitterapi_io.key.')
    out=ROOT/'data/x';raw=ROOT/'data/raw/x';out.mkdir(parents=True,exist_ok=True);raw.mkdir(parents=True,exist_ok=True)
    mode='pilot_id' if args.pilot else 'stratified_id'
    output_mode='pilot' if args.pilot else 'stratified'
    manifest_path=out/f'{output_mode}_manifest.json'
    log={'plan':plan,'requests':[],'coverage':[],'budget_cap_usd':str(args.max_usd),
        'estimated_billed_usd':'0','reserved_request_ceiling_usd':'0','completed':False}
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text(encoding='utf-8'))
        if args.refetch:
            if not previous['completed']:raise RuntimeError('Resume the incomplete run before requesting a fresh run.')
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            archive=ROOT/'tmp/x_prior_runs'/stamp;archive.mkdir(parents=True)
            shutil.copy2(manifest_path,archive/manifest_path.name)
            old_posts=out/f'{output_mode}_posts.csv'
            if old_posts.exists():shutil.copy2(old_posts,archive/old_posts.name)
            log['raw_prefix']=mode+'_'+stamp
            previous=None
        if previous is not None:
            if previous['plan']!=plan:raise RuntimeError('Existing run uses a different sampling plan; preserve it and choose a separate output run.')
            log=previous;log['budget_cap_usd']=str(args.max_usd)
            if log['completed']:
                if not (out/f'{output_mode}_posts.csv').exists():raise RuntimeError('Post bodies are not distributed. Explicit --refetch and a newly authorized --max-usd are required for paid text recollection.')
                print('Completed cached run; no new requests.');return
    mode=log.get('raw_prefix',mode)
    rows={};reserved=Decimal(log['reserved_request_ceiling_usd']);estimated=Decimal(log['estimated_billed_usd'])
    posts_path=out/f'{output_mode}_posts.csv'
    if posts_path.exists() and not args.refetch:
        with posts_path.open(encoding='utf-8-sig',newline='') as handle:
            rows={row['id']:row for row in csv.DictReader(handle)}
        backfill_cached_metadata(rows,log)
    done={row['window_id'] for row in log['coverage']}
    opener=urllib.request.build_opener(NoRedirect())
    last_request=0.0
    def save():
        log['estimated_billed_usd']=str(estimated);log['reserved_request_ceiling_usd']=str(reserved)
        if rows:
            path=out/f'{output_mode}_posts.csv'
            temporary=path.with_suffix('.csv.tmp')
            with temporary.open('w',encoding='utf-8-sig',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(next(iter(rows.values()))));writer.writeheader()
                writer.writerows(sorted(rows.values(),key=lambda row:(row['created_at'],row['id'])))
            temporary.replace(path)
        temporary=manifest_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(log,indent=2),encoding='utf-8')
        temporary.replace(manifest_path)
    try:
        for window in schedule:
            if window['id'] in done:continue
            cursor='';seen_cursors=set();complete=False;returned=0
            for page in range(args.max_pages_per_window):
                first,last=id_bounds(window)
                params={'query':f"{base_query} since_id:{first} max_id:{last}",
                    'queryType':'Latest','cursor':cursor}
                url=ENDPOINT+'?'+urllib.parse.urlencode(params)
                path=raw/f'{mode}_{window["id"]}_p{page}.json'
                relative=str(path.relative_to(ROOT)).replace('\\','/')
                prior=next((entry for entry in log['requests'] if entry['file']==relative),None)
                if path.exists():
                    payload=path.read_bytes()
                    if prior and hashlib.sha256(payload).hexdigest()!=prior['sha256']:raise RuntimeError('Cached response hash mismatch.')
                else:
                    if reserved+PAGE_RESERVATION>args.max_usd:
                        raise RuntimeError('Configured charge ceiling reached; partial results preserved.')
                    request=urllib.request.Request(url,headers={'X-API-Key':key,'User-Agent':'Academic AI sentiment research','Cache-Control':'no-cache'})
                    delay=args.request_interval-(time.monotonic()-last_request)
                    if delay>0:time.sleep(delay)
                    last_request=time.monotonic()
                    reserved+=PAGE_RESERVATION;save()
                    try:
                        with opener.open(request,timeout=35) as response:payload=response.read()
                    except urllib.error.HTTPError as error:
                        detail=error.read(2000).decode('utf-8',errors='replace').replace(key,'[REDACTED]')
                        log.setdefault('errors',[]).append({'status':error.code,'message':detail,
                            'retry_after':error.headers.get('Retry-After'),'utc':datetime.now(timezone.utc).isoformat()})
                        save()
                        raise RuntimeError(f'Provider HTTP {error.code}: {detail}; no automatic paid retries.') from None
                    path.write_bytes(payload)
                obj=json.loads(payload)
                posts=obj.get('tweets')
                if not isinstance(posts,list):raise RuntimeError('Unexpected provider response; saved locally for inspection.')
                if len(posts)>20:raise RuntimeError('Response exceeds the reserved per-page billing bound; stop and reconcile costs.')
                returned+=len(posts)
                if prior is None:
                    estimated+=RATE*max(1,len(posts))
                    log['requests'].append({'window_id':window['id'],'page':page,'url':url,
                        'retrieved_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),'returned':len(posts),
                        'sha256':hashlib.sha256(payload).hexdigest(),'file':relative})
                for post in posts:
                    created=parse_time(post['createdAt']);local=created.astimezone(NY)
                    if not window['start_epoch']<=created.timestamp()<window['end_epoch']:continue
                    if not START<=local<END:continue
                    if post.get('retweeted_tweet') or post.get('lang') not in [None,'en']:continue
                    author=post.get('author') or {};identifier=str(post['id'])
                    rows[identifier]={'id':identifier,'date':local.date().isoformat(),'created_at':local.isoformat(),
                        'author_id':str(author.get('id','')),'author':author.get('userName',''),
                        'text':post.get('text',''),'url':post.get('url') or f'https://x.com/i/status/{identifier}',
                        'conversation_id':str(post.get('conversationId','')),'is_reply':bool(post.get('isReply')),
                        'likes_at_retrieval':post.get('likeCount'),'reposts_at_retrieval':post.get('retweetCount'),
                        'views_at_retrieval':post.get('viewCount'),'author_automated_flag':author.get('isAutomated'),
                        'author_followers_at_retrieval':author.get('followers'),
                        'metrics_retrieved_utc':next(entry['retrieved_utc'] for entry in log['requests'] if entry['file']==relative),
                        'sample_window':window['id']}
                more=bool(obj.get('has_next_page'));next_cursor=obj.get('next_cursor')
                save()
                if not more:complete=True;break
                if not next_cursor or next_cursor in seen_cursors:raise RuntimeError('Pagination is inconsistent; do not treat it as complete.')
                seen_cursors.add(next_cursor);cursor=next_cursor
                time.sleep(1)
            log['coverage'].append({'window_id':window['id'],'returned':returned,'exhausted':complete,
                'capped':not complete,'pages':page+1})
            print(window['id'],'returned',returned,'unique retained',len(rows),'estimated USD',estimated,flush=True)
            save();time.sleep(1)
        log['completed']=True;save()
    finally:save()
    print('Saved',len(rows),'unique posts. Capped windows:',sum(r['capped'] for r in log['coverage']))

if __name__=='__main__':main()
