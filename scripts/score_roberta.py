"""Cross-check every record using frozen, local, quantized Twitter RoBERTa."""
from pathlib import Path
import sys,json,re,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/python_deps'))
import numpy as np,pandas as pd
import onnxruntime as ort
from tokenizers import Tokenizer
from scipy.special import softmax
MODEL=ROOT/'tmp/roberta';OUT=ROOT/'analysis'
PREPROCESSING_VERSION='url_http_and_mention_user_v1'

def normalize_text(text):
 return re.sub(r'(?<!\w)@\w+','@user',re.sub(r'https?://\S+','http',text))
def main():
 tok=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
 tok.enable_truncation(max_length=256)
 tok.enable_padding(pad_id=1,pad_token='<pad>')
 opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
 sess=ort.InferenceSession(str(MODEL/'onnx/model_quantized.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
 print('INPUTS',[(x.name,x.type) for x in sess.get_inputs()],flush=True)
 labels=['negative','neutral','positive']
 metrics_path=OUT/'roberta_inference.json'
 allmetrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
 names=['news_all','social_unfiltered','social'] if (OUT/'social_unfiltered_scored.csv').exists() else ['news_all','social']
 for name in names:
  path=OUT/f'{name}_roberta.csv'
  d=pd.read_csv(OUT/f'{name}_scored.csv',dtype={'id':str})
  if path.exists():
   cached=pd.read_csv(path,dtype={'id':str})
   if len(cached)==len(d) and cached.id.equals(d.id) and cached.text.equals(d.text) and allmetrics.get(name,{}).get('preprocessing_version')==PREPROCESSING_VERSION:
    print('CACHED',name,flush=True);continue
   print('SOURCE CHANGED; recomputing',name,flush=True)
  texts=d.text.map(normalize_text).tolist()
  if name=='social' and (OUT/'social_unfiltered_roberta.csv').exists():
   full=pd.read_csv(OUT/'social_unfiltered_roberta.csv',dtype={'id':str}).set_index('id')
   if set(d.id).issubset(full.index) and full.loc[d.id,'text'].tolist()==d.text.tolist():
    columns=[c for c in full if c.startswith('roberta_')]
    for col in columns:d[col]=full.loc[d.id,col].to_numpy()
    d.to_csv(path,index=False,encoding='utf-8-sig')
    allmetrics[name]={'n':len(d),'truncated_at_256_tokens':int(d.roberta_truncated.sum()),'preprocessing_version':PREPROCESSING_VERSION,'prediction_source':'Matched unfiltered X predictions; no duplicate inference.'}
    print('REUSED exact X subset predictions',name,len(d),flush=True);continue
  order=np.argsort([len(s) for s in texts]);scores=np.empty((len(d),3));truncated=0;truncated_flags=np.zeros(len(d),dtype=bool);t0=time.time()
  for start in range(0,len(order),32):
   idx=order[start:start+32];enc=tok.encode_batch([texts[i] for i in idx])
   truncated+=sum(bool(e.overflowing) for e in enc)
   truncated_flags[idx]=[bool(e.overflowing) for e in enc]
   feed={'input_ids':np.array([e.ids for e in enc],dtype=np.int64),'attention_mask':np.array([e.attention_mask for e in enc],dtype=np.int64)}
   scores[idx]=softmax(sess.run(None,feed)[0],axis=1)
   if start%1024==0:print(name,start,len(d),'seconds',round(time.time()-t0,1),flush=True)
  for i,l in enumerate(labels):d['roberta_'+l]=scores[:,i]
  d['roberta_score']=scores[:,2]-scores[:,0];d['roberta_tone']=[labels[i] for i in scores.argmax(axis=1)]
  d['roberta_truncated']=truncated_flags
  d.to_csv(path,index=False,encoding='utf-8-sig')
  allmetrics[name]={'n':len(d),'truncated_at_256_tokens':truncated,'seconds':time.time()-t0,'preprocessing_version':PREPROCESSING_VERSION}
  print('DONE',name,allmetrics[name],flush=True)
 metrics_path.write_text(json.dumps(allmetrics,indent=2),encoding='utf-8')
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
