import os
os.environ['USE_LIBUV']='0'
import importlib.util,json,socket,sys,tempfile
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
ROOT=Path(__file__).resolve().parent

def worker(rank,port,tmp):
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port[0]),WORLD_SIZE='2',RANK=str(rank),LOCAL_RANK=str(rank))
    jobs=json.loads((ROOT/'manifest.json').read_text())['jobs']
    for idx,opt in enumerate(['sgd','polyak','nesterov']):
        j=next(j for j in jobs if j['optimizer']==opt)
        spec=importlib.util.spec_from_file_location('entry_'+opt,ROOT/j['script']);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
        flags=['--device','cpu','--train',str(Path(tmp)/'data.bin'),'--val',str(Path(tmp)/'data.bin'),'--output',str(Path(tmp)/opt),'--vocab-size','32','--width','16','--layers','1','--heads','2','--context','4','--validation-contexts','4','--validation-batch-size','2','--microbatch','64','--eval-every','20','--checkpoint-every','20','--loss-chunk-tokens','8']
        os.environ['MASTER_PORT']=str(port[idx*2])
        print('entry',rank,opt,flush=True)
        m.run(m.parser().parse_args(flags))
        # A completed checkpoint must resume without extra updates/duplicate log rows.
        os.environ['MASTER_PORT']=str(port[idx*2+1])
        print('resume',rank,opt,flush=True)
        m.run(m.parser().parse_args(flags))
        if rank==0:
            out=Path(tmp)/opt;cfg=json.loads((out/'config.json').read_text());state=torch.load(out/'latest.pt',weights_only=False)
            steps=[json.loads(x) for x in (out/'steps.jsonl').read_text().splitlines()]
            metrics=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
            assert len(steps)==cfg['updates'] and len({x['updates'] for x in metrics})==len(metrics)
            assert steps[-1]['tokens']==cfg['target_train_tokens'] and state['position']==cfg['target_train_tokens']
            assert cfg['world_size']==2 and cfg['microbatch']==64
            assert not any(k.startswith('module.') for k in state['model'])
    if rank==0:(ROOT/'entry_verification.json').write_text(json.dumps(dict(cpu_two_process_gloo=True,optimizers=['sgd','polyak','nesterov'],real_entry_tiny_models=True,full_budget_and_partial_tail=True,single_writer_logs=True,final_checkpoint_resume=True,cloud_nccl_full_model=False),indent=2))

if __name__=='__main__':
    sockets=[socket.socket() for _ in range(6)]
    for sock in sockets:sock.bind(('127.0.0.1',0))
    port=[sock.getsockname()[1] for sock in sockets]
    for sock in sockets:sock.close()
    with tempfile.TemporaryDirectory() as tmp:
        with (Path(tmp)/'data.bin').open('wb') as f:
            h=np.zeros(256,dtype=np.int32);h[:3]=[20240520,1,200000];h.tofile(f);(np.arange(200000,dtype=np.int64)%32).astype(np.uint16).tofile(f)
        mp.spawn(worker,args=(port,tmp),nprocs=2,join=True)
    print('Actual two-rank entry, logs, validation, full-budget checkpoint and resume passed.')
