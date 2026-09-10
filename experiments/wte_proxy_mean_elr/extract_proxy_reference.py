"""Extract proxy targets from one completed baseline stage of wte_mean_elr_pipeline."""
import argparse,hashlib,json,math,os
from pathlib import Path

def canonical(name):
    while name.startswith(('module.','_orig_mod.')):name=name.split('.',1)[1]
    return name

def extract(baseline,output):
    baseline=Path(baseline);output=Path(output)
    cfg=json.loads((baseline/'config.json').read_text(encoding='utf-8'))
    done=json.loads((baseline/'complete.json').read_text(encoding='utf-8'))
    assert cfg['role']=='baseline' and done['config']==cfg and done['completed_step']==cfg['num_iterations']
    log=(baseline/'train.log').read_text(encoding='utf-8')
    assert f"step:{cfg['num_iterations']}/{cfg['num_iterations']} val_loss:" in log
    source=baseline/'dense_mean_elr.jsonl';count=6*cfg['n_layer']+1
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(output)+'.tmp');canonical_names=None;rows=0;source_hash=hashlib.sha256()
    with source.open('rb') as inp,tmp.open('w',encoding='utf-8') as out:
        for raw in inp:
            source_hash.update(raw);r=json.loads(raw);rows+=1
            assert r['update_step']==rows and r['pre_update_step']==rows-1
            names=[canonical(n) for n in r['names']];rms=r['rms']
            assert len(names)==len(rms)==count and len(set(names))==count
            assert names.count('transformer.wte.weight')==1
            assert all(n=='transformer.wte.weight' or n.startswith('transformer.h.') for n in names)
            assert all(math.isfinite(v) and v>0 for v in rms)
            pairs=sorted(zip(names,rms),key=lambda nv:(nv[0]=='transformer.wte.weight',nv[0]))
            names=[n for n,v in pairs];rms=[v for n,v in pairs]
            if canonical_names is None:canonical_names=names
            assert names==canonical_names
            lr=r['block_lr'];assert math.isfinite(lr) and lr>0
            assert math.isclose(r['embed_lr'],2*lr,rel_tol=1e-10)
            assert r['weight_decay']==cfg['weight_decay']
            proxy=sum(lr/v for v in rms)/count
            actual=(sum(lr/v for v in rms[:-1])+2*lr/rms[-1])/count
            assert math.isclose(actual,r['actual_mean_elr'],rel_tol=1e-8)
            out.write(json.dumps(dict(update_step=rows,pre_update_step=rows-1,names=names,
                proxy_mean_elr=proxy,actual_mean_elr=actual,reference_block_lr=lr,
                definition='mean_i(block_lr / RMS(W_i)); WTE numerator uses block_lr; actual WTE LR remains 2x'))+'\n')
    assert rows==cfg['num_iterations']
    os.replace(tmp,output)
    metadata=dict(source=str(source.resolve()),source_sha256=source_hash.hexdigest(),rows=rows,controlled_tensors=count,baseline_config=cfg)
    Path(str(output)+'.metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return cfg

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();extract(a.baseline,a.output);print(a.output)
