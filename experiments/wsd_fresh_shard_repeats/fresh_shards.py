"""Use the untouched suffix of the known sequential trainer; metadata only."""
KNOWN_SHARED_SOURCES={'92b8b365e80a7d738bd292014f9adbb67bc515358175b50060ad81956faac6c1','bf1ba81bad94b4dad48b10d441730286d10f7f1bf456f02f5db5ee8c32930e37'}
KNOWN_SHARED_SCHEDULE='9e3a4b48e9c2de544e2c1c7f35775c80f910b39717eed85ac45645434961c387'

def audit_history(state):
    # The pinned trainer reads shards sequentially from index0. Skip its entire
    # current shard, including any partially used/prefetched data. No replay/I/O.
    if state['source_sha256'] not in KNOWN_SHARED_SOURCES or state['schedule_sha256']!=KNOWN_SHARED_SCHEDULE or state['step']!=16000 or state.get('probe_experiment')!='stable':
        raise ValueError('requires the known shared-stable16000 checkpoint')
    loader=state['train_loader'];files=loader['files'];current=loader['shard']
    B,T,world=(loader[k] for k in ('B','T','world'))
    width=B*T*world;accum=state['config']['batch_size']//(B*world)
    counts=[(size-1024)//2 for size in loader['sizes']]
    calls=16000*accum+1
    # Cheap metadata-only check: after a full corpus pass no shard is untouched.
    if calls>=sum((count-1)//width for count in counts):
        raise ValueError('training reached a full corpus pass; no untouched suffix')
    if not 0<=current<len(files):raise ValueError('invalid current shard index')
    return dict(mode='checkpoint suffix, no historical replay or shard scan',
                fork_step=16000,fetches_including_prefetch=calls,
                gradient_accumulation=accum,global_microbatch_tokens=width,
                files=files,token_counts=counts,excluded_indices=list(range(current+1)),
                unvisited_indices=list(range(current+1,len(files))),
                saved_shard=current,saved_position=loader['position'])

def check_fresh_shard(state,index,updates):
    audit=audit_history(state)
    if index not in audit['unvisited_indices']:raise ValueError('selected shard is not after the checkpoint current shard')
    return audit,dict(index=index,path=audit['files'][index],tokens=audit['token_counts'][index])
