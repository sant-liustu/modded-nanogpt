"""Extract block-only weighted Muon ELR and independent AdamH embedding ELR."""
import argparse, hashlib, json, math, os
from pathlib import Path

def canonical(name):
    while name.startswith(('module.','_orig_mod.')): name=name.split('.',1)[1]
    return name

def extract(baseline, output, steps=20400):
    baseline=Path(baseline); output=Path(output)
    if output.exists() or Path(str(output)+'.metadata.json').exists(): raise FileExistsError(output)
    logs=list(baseline.glob('*.txt'))
    if baseline.with_suffix('.txt').exists(): logs.append(baseline.with_suffix('.txt'))
    if not any(f'step:{steps}/{steps} val_loss:' in p.read_text(encoding='utf-8') for p in logs):
        raise ValueError('Missing completed baseline text log')
    meta_bytes=(baseline/'tensor_metadata.json').read_bytes()
    meta={}
    for r in json.loads(meta_bytes):
        n=canonical(r['name'])
        if r['ndim']!=2 or not (n.startswith('transformer.h.') or n=='transformer.wte.weight'):continue
        if n in meta or r['numel']!=math.prod(r['shape']) or r['numel']<=0: raise ValueError('Invalid tensor metadata')
        meta[n]=r
    names=sorted(meta); numels=[meta[n]['numel'] for n in names]; N=sum(numels)
    layers={n.split('.')[2] for n in names if n.startswith('transformer.h.')}
    components=['attn.c_q.weight','attn.c_k.weight','attn.c_v.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']
    expected={f'transformer.h.{i}.{c}' for i in range(len(layers)) for c in components}|{'transformer.wte.weight'}
    if set(names)!=expected: raise ValueError('Unexpected controlled tensor set')
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(str(output)+'.tmp'); source=baseline/'muonhyperball_norm_history.jsonl'
    digest=hashlib.sha256(); step=1; entries={}; rows=0
    try:
        with source.open('rb') as inp,temp.open('w',encoding='utf-8') as out:
            def flush():
                if set(entries)!=set(names): raise ValueError(f'Incomplete step {step}')
                block_lrs=[entries[n]['lr'] for n in names if n!='transformer.wte.weight']
                base_lr=block_lrs[0]
                if not math.isfinite(base_lr) or base_lr<=0 or not all(math.isclose(v,base_lr,rel_tol=1e-12) for v in block_lrs):
                    raise ValueError('Reference does not have one base block LR')
                rms=[entries[n]['raw_update_fro_norm']/math.sqrt(meta[n]['numel']) for n in names]
                if not all(math.isfinite(v) and v>0 for v in rms): raise ValueError('Invalid raw update RMS')
                ei=names.index('transformer.wte.weight')
                block_numel=N-numels[ei]
                block_target=math.fsum(k*base_lr/v for i,(k,v) in enumerate(zip(numels,rms)) if i!=ei)/block_numel
                embedding_lr=entries['transformer.wte.weight']['lr']
                if not math.isfinite(embedding_lr) or embedding_lr<=0: raise ValueError('Invalid embedding LR')
                embedding_target=embedding_lr/rms[ei]
                out.write(json.dumps(dict(update_step=step,pre_update_step=step-1,block_target=block_target,embedding_target=embedding_target,
                    reference_base_lr=base_lr,reference_embedding_lr=entries['transformer.wte.weight']['lr']))+'\n')
            for raw in inp:
                digest.update(raw); r=json.loads(raw); n=canonical(r['name'])
                if n not in meta: continue
                if r['step']!=step:
                    if r['step']!=step+1: raise ValueError('Non-consecutive steps')
                    flush();rows+=1;entries={};step=r['step']
                if n in entries: raise ValueError('Duplicate tensor row')
                if list(r['shape'])!=meta[n]['shape'] or not r.get('has_grad',True): raise ValueError('Invalid shape/gradient')
                entries[n]=r
            flush();rows+=1
        if rows!=steps or step!=steps: raise ValueError(f'Expected {steps} steps, found {rows}')
        metadata=dict(metric='block_weighted_and_embedding_raw_update_rms',names=names,numels=numels,
            total_controlled_numel=N,block_numel=N-meta["transformer.wte.weight"]["numel"],rows=rows,source=str(source.resolve()),source_sha256=digest.hexdigest(),
            tensor_metadata_sha256=hashlib.sha256(meta_bytes).hexdigest(),
            definition='block_target=sum_blocks(numel_i/N_blocks * lr_i/RMS(U_i)); embedding_target=embedding_lr/RMS(U_embedding); gamma excluded; no interpolation')
        meta_tmp=Path(str(output)+'.metadata.json.tmp');meta_tmp.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        os.replace(temp,output);os.replace(meta_tmp,Path(str(output)+'.metadata.json'))
        return metadata
    except BaseException:
        temp.unlink(missing_ok=True)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--steps',type=int,default=20400)
    a=p.parse_args();m=extract(a.baseline,a.output,a.steps);print(f"Extracted {m['rows']} steps, {len(m['names'])} matrices: {a.output}")
