"""One-command formal MuonW WD0.1 run using the bundled real Hyperball reference."""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', help='Two visible GPU IDs; defaults to CUDA_VISIBLE_DEVICES or 0,1')
    p.add_argument('--data-dir', type=Path, default=REPO/'data/fineweb10B')
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true', help='Validate bundled reference and print command without training')
    a=p.parse_args()
    ids=(a.gpus if a.gpus is not None else os.environ.get('CUDA_VISIBLE_DEVICES','0,1')).split(',')
    ids=[x.strip() for x in ids]
    if len(ids)!=2 or len(set(ids))!=2 or not all(ids): p.error('Exactly two distinct GPUs are required')
    reference=HERE/'hyperball_weighted_proxy_reference.jsonl'
    manifest=json.loads((HERE/'reference_sha256.json').read_text(encoding='utf-8'))
    for name,expected in manifest.items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Bundled reference checksum mismatch: '+name)
    metadata=json.loads(Path(str(reference)+'.metadata.json').read_text(encoding='utf-8'))
    assert metadata['rows']==20400 and len(metadata['names'])==73
    assert metadata['metric']=='weighted_base_lr_over_raw_update_rms'
    data=a.data_dir.resolve()
    cfg={'input_bin':str(data/'fineweb_train_*.bin'),'input_val_bin':str(data/'fineweb_val_*.bin')}
    out=a.output.resolve() if a.output else REPO/'logs'/('hyperball_weighted_muonw_wd0p1_'+uuid.uuid4().hex)
    if out.exists(): raise FileExistsError(out)
    config=out.parent/(out.name+'.launch.json')
    env=os.environ.copy()
    for key in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','ROLE_RANK','TORCHELASTIC_RUN_ID']:
        env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=','.join(ids),PYTHONUNBUFFERED='1')
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2','--max-restarts=0',
         str(HERE/'train_muonw_wd0p1.py'),'--reference',str(reference),'--output',str(out),'--config',str(config)]
    print(json.dumps(dict(gpus=ids,reference=str(reference),reference_steps=20400,output=str(out),config=cfg,command=cmd),indent=2),flush=True)
    if a.dry_run:return
    for pattern in cfg.values():
        if not glob.glob(pattern):raise FileNotFoundError('No dataset shards: '+pattern)
    out.parent.mkdir(parents=True,exist_ok=True)
    with config.open('x',encoding='utf-8') as f:json.dump(cfg,f,indent=2)
    subprocess.run(cmd,env=env,cwd=REPO,check=True)

if __name__=='__main__':main()
