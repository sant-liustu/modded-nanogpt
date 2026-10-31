"""Two CPU/Gloo ranks: LLaMA Fisher serial reference and actual training entry."""
import json
import os
from pathlib import Path
import socket
import sys
import time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import train_llama025b_fineweb as run


def fisher_worker(rank, url):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=url,rank=rank,world_size=2,timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(67)
        model=run.Llama(run.LlamaConfig(41,32,128,1,4))
        x=torch.randint(41,(8,4))
        y=torch.zeros_like(x)
        y[1,:]=-1
        y[6,1:]=-1
        part=slice(rank*4,(rank+1)*4)
        record=run.probe(model,x[part],y[part],2,103,run.parameter_rms(model),2,True,3)
        assert record['draw_seeds_by_rank']==[[103,104],[105,106]]
        assert record['world_size']==2 and record['sequences']==8 and record['local_sequences']==4
        worst=0.0
        for draw in range(2):
            objective=0
            for r in range(2):
                generator=torch.Generator().manual_seed(103+draw*2+r)
                for b in range(r*4,(r+1)*4,2):
                    hidden=model(x[b:b+2],return_hidden=True).reshape(-1,32)
                    mask=y[b:b+2].reshape(-1)!=-1
                    for t in range(0,len(hidden),3):
                        valid=mask[t:t+3]
                        if not valid.any(): continue
                        logits=model.lm_head(hidden[t:t+3])
                        labels=torch.multinomial(logits.detach().softmax(-1),1,generator=generator).squeeze(-1)
                        labels[~valid]=-1
                        objective+=run.F.cross_entropy(logits,labels,ignore_index=-1,reduction='sum')/record['valid_tokens']
            gradients=torch.autograd.grad(objective,tuple(model.parameters()))
            for (name,v),g in zip(model.named_parameters(),gradients):
                expected=record['valid_tokens']*g.double().square().sum().item()/v.numel()
                actual=record['parameters'][name]['mean_fisher_draws'][draw]
                error=abs(actual-expected)/max(expected,1e-15)
                worst=max(worst,error)
                assert error<2e-5,(name,error)
        shared=[None,None]
        dist.all_gather_object(shared,record['groups'])
        assert shared[0]==shared[1]
        if rank==0: print('PASS two-rank LLaMA Fisher SUM-before-square, serial objective; max relative error',worst,flush=True)
    finally: dist.destroy_process_group()


def entry_worker(rank, port, output, data):
    os.environ.update(RANK=str(rank),WORLD_SIZE='2',LOCAL_RANK=str(rank),MASTER_ADDR='127.0.0.1',
                      MASTER_PORT=str(port),USE_LIBUV='0')
    sys.argv=['train_llama025b_fineweb.py','--input-bin',data,'--input-val-bin',data,
              '--output',output,'--device','cpu','--hidden-size','32','--intermediate-size','128',
              '--num-layers','1','--num-heads','4','--vocab-size','41','--sequence-length','4',
              '--batch-size','4','--micro-batch-size','1','--stop-step','2','--warmup-steps','1',
              '--fisher-sequences','4','--fisher-micro-batch-size','1','--fisher-token-chunk-size','3',
              '--fisher-every','1','--save-every','1','--val-every','1','--val-batches','1']
    run.main()


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        return s.getsockname()[1]


if __name__=='__main__':
    out=Path(__file__).parent/'local_checks'/('paper_llama_ddp_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True)
    data=out/'synthetic.bin'
    tokens=np.arange(1000,dtype=np.uint16)%41
    header=np.zeros(256,dtype=np.int32); header[:3]=[20240520,1,len(tokens)]
    with open(data,'wb') as f: f.write(header.tobytes()); f.write(tokens.tobytes())
    mp.spawn(fisher_worker,args=(f'tcp://127.0.0.1:{port()}?use_libuv=0',),nprocs=2,join=True)
    mp.spawn(entry_worker,args=(port(),str(out/'run'),str(data)),nprocs=2,join=True)
    ckpt=out/'run'/'checkpoints'/'step000002'
    states=[torch.load(ckpt/f'rank{r}.pt',weights_only=False) for r in range(2)]
    assert json.loads((ckpt/'complete.json').read_text())==dict(step=2,world_size=2)
    for k in states[0]['model']: assert torch.equal(states[0]['model'][k],states[1]['model'][k])
    # Compare ACTUAL entry-point DDP+accumulation updates with a serial global-batch AdamW.
    torch.set_num_threads(1)
    reference=run.Llama(run.LlamaConfig(41,32,128,1,4))
    initial=torch.load(out/'run'/'fisher_models'/'step000000.pt',weights_only=False)
    reference.load_state_dict(initial['model'])
    groups=[{'params':[v for n,v in reference.named_parameters() if run.block_type(n)==name],'name':name}
            for name in ('Emb','Head','Norm','QK','VO','FFN')]
    optimizer=torch.optim.AdamW(groups,lr=8e-4,betas=(0.9,0.95),eps=1e-8,weight_decay=0.1,fused=False)
    samplers=[run.TokenData(str(data),4,41,r,2) for r in range(2)]
    for iteration in range(2):
        batches=[sampler.batch(1) for sampler in samplers for _ in range(2)]
        x=torch.cat([b[0] for b in batches]); y=torch.cat([b[1] for b in batches])
        optimizer.zero_grad(set_to_none=True)
        _,loss=reference(x,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.0)
        for group in optimizer.param_groups: group['lr']=run.learning_rate(iteration,8e-4,1,50000)
        optimizer.step()
    worst=max((v-states[0]['model'][name]).abs().max().item() for name,v in reference.state_dict().items())
    for name,v in reference.state_dict().items():
        torch.testing.assert_close(v,states[0]['model'][name],rtol=2e-5,atol=2e-7)
    assert states[0]['data_state']['position']==states[1]['data_state']['position']
    assert [s['data_state']['rank'] for s in states]==[0,1]
    rows=[json.loads(s) for s in (out/'run'/'fisher_history.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==[0,1,2] and all(r['world_size']==2 for r in rows)
    grads=[json.loads(s) for s in (out/'run'/'tensor_grad_norm_history.jsonl').read_text().splitlines()]
    assert [g['step'] for g in grads]==[1,2]
    for row in grads:
        assert set(row['parameters'])==set(states[0]['model'])
        recovered=sum(v['grad_norm']**2 for v in row['parameters'].values())**0.5
        assert abs(recovered-row['global_grad_norm'])<1e-5
    (out/'report.json').write_text(json.dumps(dict(two_process_gloo=True,serial_fisher_agreement=True,
                                                actual_entry_training=True,steps=2,rank_weights_equal=True,
                                                serial_AdamW_global_batch_max_weight_error=worst)))
    print('PASS actual two-rank training vs serial AdamW, measurement, validation, checkpoint; output',out)
