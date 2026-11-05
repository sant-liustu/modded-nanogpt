"""Audit the exact pinned sorted/cyclic training-loader history; no checkpoint bypass."""
import glob,os,struct
from pathlib import Path
KNOWN_SHARED_SOURCES={'92b8b365e80a7d738bd292014f9adbb67bc515358175b50060ad81956faac6c1','bf1ba81bad94b4dad48b10d441730286d10f7f1bf456f02f5db5ee8c32930e37'}
KNOWN_SHARED_SCHEDULE='9e3a4b48e9c2de544e2c1c7f35775c80f910b39717eed85ac45645434961c387'

def shard_tokens(path):
    with open(path,'rb') as f:header=f.read(1024)
    if len(header)!=1024:raise ValueError('short shard header: '+str(path))
    magic,version,count=struct.unpack('<3i',header[:12])
    if magic!=20240520 or version!=1 or count<=0 or os.path.getsize(path)!=1024+2*count:raise ValueError('invalid shard: '+str(path))
    return count

def audit_history(state):
    if state['source_sha256'] not in KNOWN_SHARED_SOURCES or state['schedule_sha256']!=KNOWN_SHARED_SCHEDULE or state['step']!=16000 or state.get('probe_experiment')!='stable':raise ValueError('freshness proof requires pinned shared-stable16000 checkpoint')
    loader=state['train_loader'];files=loader['files'];B,T,world,rank=(loader[k] for k in ('B','T','world','rank'))
    if not files or files!=sorted(files) or len(set(files))!=len(files):raise ValueError('history is not the original sorted shard list')
    if min(B,T,world)<=0 or not 0<=rank<world:raise ValueError('invalid loader geometry')
    cfg=state['config'];batch=cfg['batch_size']
    if cfg['sequence_length']!=T or batch%(B*world):raise ValueError('invalid saved global batch/context')
    for i,path in enumerate(files):
        if str(Path(path).resolve())!=path or os.path.getsize(path)!=loader['sizes'][i] or os.stat(path).st_mtime_ns!=loader['modified_ns'][i]:raise ValueError('historical shard path/signature changed: '+path)
    counts=[shard_tokens(p) for p in files];width=B*T*world
    if any(n<width+1 for n in counts):raise ValueError('shard shorter than global microbatch')
    accum=batch//(B*world);calls=16000*accum+1
    shard=position=0;visited=set()
    for _ in range(calls):
        visited.add(shard);position+=width
        if position+width+1>counts[shard]:shard=(shard+1)%len(files);position=0
    # The original loader may preload the next shard after its last prefetched batch.
    visited.add(shard)
    if shard!=loader['shard'] or position+rank*B*T!=loader['position']:raise ValueError('cannot prove used shards: saved cursor differs from original-loader replay')
    return dict(fork_step=16000,fetches_including_prefetch=calls,gradient_accumulation=accum,global_microbatch_tokens=width,files=files,token_counts=counts,excluded_indices=sorted(visited),unvisited_indices=[i for i in range(len(files)) if i not in visited],saved_shard=shard,saved_position=loader['position'])

def check_fresh_shard(state,index,updates):
    audit=audit_history(state)
    if index not in audit['unvisited_indices']:raise ValueError('selected shard was trained, partially used, prefetched or preloaded')
    required=(updates*audit['gradient_accumulation']+1)*audit['global_microbatch_tokens']+1
    if audit['token_counts'][index]<required:raise ValueError('fresh shard cannot hold the continuation plus prefetched batch; no wrap/spill allowed')
    return audit,dict(index=index,path=audit['files'][index],tokens=audit['token_counts'][index],required_tokens=required)
