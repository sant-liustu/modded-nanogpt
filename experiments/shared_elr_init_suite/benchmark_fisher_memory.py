"""Small CUDA comparison; pass a pre-fix fisher_diagnostics.py as baseline."""
import argparse, json, time, types
from pathlib import Path
import torch
from verify_fisher_chunked import ns
from fisher_diagnostics import probe
parser=argparse.ArgumentParser()
parser.add_argument('--baseline-helper',type=Path,required=True)
parser.add_argument('--sequence-length',type=int,default=1024)
parser.add_argument('--token-chunk-size',type=int,default=1024)
a=parser.parse_args()
source=a.baseline_helper.read_text()
old=types.ModuleType('old_fisher');exec(compile(source,str(a.baseline_helper),'exec'),old.__dict__)
amp=types.ModuleType('old_fisher_amp')
exec(compile(source.replace("torch.autocast(device.type, enabled=False)","torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda)"),str(a.baseline_helper),'exec'),amp.__dict__)
torch.manual_seed(81)
m=ns['GPT'](ns['GPTConfig'](vocab_size=100278,n_layer=1,n_head=1,n_embd=128)).cuda()
x=torch.randint(100278,(2,a.sequence_length),device='cuda');y=torch.zeros_like(x)
results={}
for name,fn,kwargs in [('old_fp32',old.probe,{}),('old_bf16',amp.probe,{}),('chunked_fp32',probe,dict(token_chunk_size=a.token_chunk_size,use_bf16=False)),('chunked_bf16',probe,dict(token_chunk_size=a.token_chunk_size,use_bf16=True))]:
    torch.cuda.empty_cache()
    record=fn(m,x,y,1,53,micro_batch_size=2,**kwargs)
    results[name]={k:record[k] for k in ('peak_allocated_bytes','extra_peak_allocated_bytes','seconds')}
report=dict(scope=f'single CUDA w128/d1 vocab100278 B2 T{a.sequence_length}, token chunk{a.token_chunk_size} for streamed cases; eager; no optimizer state',results=results)
out=Path(__file__).parent/'local_checks'/('fisher_memory_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2));print(out)
assert results['chunked_bf16']['peak_allocated_bytes']<results['old_bf16']['peak_allocated_bytes']
