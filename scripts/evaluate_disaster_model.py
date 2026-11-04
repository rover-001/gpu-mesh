"""Evaluate real llama-server generations, never retrieval or canned responses."""
import argparse,hashlib,json,os,re,subprocess,time,urllib.request
from pathlib import Path

def evaluate(args):
 root=Path(__file__).resolve().parents[1]
 output=Path(args.checkpoints)
 manifest=json.loads((output/'active_adapter.json').read_text())
 adapter=output/manifest['adapter_file']
 if hashlib.sha256(adapter.read_bytes()).hexdigest()!=manifest['adapter_sha256']:raise ValueError('Adapter checksum mismatch')
 base=Path(args.model)
 url=f'http://127.0.0.1:{args.port}'
 def req(path,data=None):
  request=urllib.request.Request(url+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=120) as result:return json.load(result)
 probes=json.loads((root/'data/disaster_generation_eval.json').read_text())
 scales = [None] if args.merged else ([0,1] if args.compare_base else [1])
 report={'inference_mode':'merged_weights' if args.merged else 'base_plus_lora','base':str(base),'adapter':str(adapter),'manifest':manifest,'results':[]}
 logfile=Path(args.report).with_suffix('.server.log')
 with logfile.open('w') as log:
  env=os.environ.copy()
  if os.name!='nt':env['LD_LIBRARY_PATH']=str(root/'tools/llama-bin/llama-b12020')+':/usr/local/lib/ollama/cuda_v12:'+env.get('LD_LIBRARY_PATH','')
  process=subprocess.Popen([args.server,'-m',str(base),* ([] if args.merged else ['--lora',str(adapter)]),'--host','127.0.0.1','--port',str(args.port),'-ngl',str(args.gpu_layers),'-c','2048','-t','4','--jinja','--chat-template-file',str(root/'data/disaster_model_chat_template.jinja'),'--repeat-penalty','1.1'],stdout=log,stderr=log,env=env)
  try:
   for _ in range(120):
    if process.poll() is not None:raise RuntimeError(logfile.read_text()[-2000:])
    try:req('/health');break
    except Exception:time.sleep(1)
   report['properties']=req('/props')
   for scale in scales:
    if scale is not None: req('/lora-adapters',[{'id':0,'scale':scale}])
    for probe in probes:
     body={'model':'disaster-1.5b','messages':[{'role':'system','content':'You are a helpful assistant. Answer clearly and briefly.'},{'role':'user','content':probe['prompt']}],'temperature':0,'seed':42,'max_tokens':160,'stream':False,'cache_prompt':False}
     result=req('/v1/chat/completions',body)
     message=result['choices'][0]['message'];text=message.get('content') or ''
     passed=all(re.search(pattern,text,re.I) for pattern in probe['required_patterns']) and not message.get('tool_calls') and '<tool_call>' not in text
     row={'adapter_scale':scale,'prompt':probe['prompt'],'answer':text,'passed':bool(passed)}
     report['results'].append(row);print(json.dumps(row),flush=True)
   report['scores']={str(scale):{'passed':sum(r['passed'] for r in report['results'] if r['adapter_scale']==scale),'total':sum(r['adapter_scale']==scale for r in report['results'])} for scale in scales}
   Path(args.report).write_text(json.dumps(report,indent=2));print(json.dumps(report['scores']),flush=True)
  finally:
   process.terminate()
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:process.kill();process.wait()
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--server',required=True);p.add_argument('--model',required=True);p.add_argument('--checkpoints',required=True);p.add_argument('--report',required=True);p.add_argument('--port',type=int,default=18185);p.add_argument('--gpu-layers',type=int,default=99);p.add_argument('--compare-base',action='store_true');p.add_argument('--merged',action='store_true');evaluate(p.parse_args())
