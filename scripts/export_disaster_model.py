"""Merge a real learned PEFT adapter and quantize the resulting whole model."""
import argparse,hashlib,json,os,subprocess,tempfile
from pathlib import Path

def export(checkpoints, quantizer=None):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 from train_qwen import read_checkpoint, atomic_json
 root=Path(__file__).resolve().parents[1]
 folder,metadata=read_checkpoint(Path(checkpoints)/'best.json')
 destination=folder/'model-q8_0.gguf'
 if not destination.exists():
  base=Path(metadata['base_model_path'])
  if not base.is_dir():base=(Path('C:/rgpu/training-base') if os.name=='nt' else Path.home()/'.cache/gpu-mesh/training-base')/'Qwen2.5-1.5B-Instruct'
  with tempfile.TemporaryDirectory(prefix='rgpu-merge-') as temp:
   temp=Path(temp);model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16)
   model=PeftModel.from_pretrained(model,folder).merge_and_unload(safe_merge=True)
   model.save_pretrained(temp/'hf',safe_serialization=True);AutoTokenizer.from_pretrained(base).save_pretrained(temp/'hf')
   del model
   import gc;gc.collect()
   subprocess.run([os.sys.executable,str(root/'tools/llama.cpp/convert_hf_to_gguf.py'),str(temp/'hf'),'--outfile',str(temp/'merged-f16.gguf'),'--outtype','f16'],check=True)
   executable=quantizer or str(root/('tools/llama-quantize.exe' if os.name=='nt' else 'tools/llama-bin/llama-b12020/llama-quantize'))
   env=os.environ.copy();env['LD_LIBRARY_PATH']=str(Path(executable).parent)+':/usr/local/lib/ollama/cuda_v12:'+env.get('LD_LIBRARY_PATH','')
   subprocess.run([executable,str(temp/'merged-f16.gguf'),str(destination),'Q8_0'],check=True,env=env)
  if destination.open('rb').read(4)!=b'GGUF':raise ValueError('Invalid merged model')
 digest=hashlib.sha256()
 with destination.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
 record={'name':'Qwen 2.5 1.5B Disaster (Trained)','hf_repo':'','filename':str(Path('checkpoints-disaster-1.5b')/folder.name/destination.name).replace('\\','/'),'sha256':digest.hexdigest(),'size':destination.stat().st_size}
 atomic_json(Path(checkpoints)/'candidate_model.json',record)
 print(json.dumps(record),flush=True)
 return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoints',required=True);p.add_argument('--quantizer');a=p.parse_args();export(a.checkpoints,a.quantizer)
