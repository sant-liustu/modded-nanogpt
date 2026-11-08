"""Independent projected-position Nesterov reference, including clipping and resume."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parent


class Quadratic(torch.nn.Module):
    def __init__(self,device):
        super().__init__()
        self.matrix=torch.nn.Parameter(torch.tensor([[.8,-.3],[.2,.4]],device=device))
        self.gamma=torch.nn.Parameter(torch.tensor([1.,.7],device=device))
        self.seen=[]

    def forward(self,x,y,return_logits=False):
        self.seen.append([p.detach().clone() for p in self.parameters()])
        h=self.matrix.new_tensor([[100.,400.],[200.,300.]])
        q=self.gamma.new_tensor([20.,70.])
        return None, .5*(h*self.matrix.square()).sum()+.5*(q*self.gamma.square()).sum()


class Tape:
    def batch(self,position,size,device):
        return torch.zeros(size,1,device=device),torch.zeros(size,1,device=device)


def check(module,beta,device):
    model=Quadratic(device)
    args=SimpleNamespace(lr=.01,momentum=beta,arm='fixed',context=1,batch=4,
                         microbatch=1,target_train_tokens=100,device=device)
    optimizer=module.build_optimizer(model,args)
    targets=module.targets_for(model)
    expected=[p.detach().clone() for p in model.parameters()]
    previous=[p.clone() for p in expected]
    old_velocity=[torch.zeros_like(p) for p in expected]
    max_y_gap_from_raw_velocity=0.
    for index,eta in enumerate((.01,.02,.007,.03)):
        base=[p.clone() for p in expected]
        d=[p-q for p,q in zip(base,previous)]
        lookahead=[p+beta*v for p,v in zip(base,d)]
        if index:
            gap=max(float((beta*(a-b)).abs().max()) for a,b in zip(d,old_velocity))
            max_y_gap_from_raw_velocity=max(max_y_gap_from_raw_velocity,gap)
        h=base[0].new_tensor([[100.,400.],[200.,300.]])
        q=base[1].new_tensor([20.,70.])
        grads=[h*lookahead[0],q*lookahead[1]]
        norm=torch.stack([g.square().sum() for g in grads]).sum().sqrt()
        coefficient=(50/(norm+1e-6)).clamp(max=1.)
        grads=[g*coefficient for g in grads]
        delta=[beta*v-eta*g for v,g in zip(d,grads)]
        expected=[p+v for p,v in zip(base,delta)]
        expected[0].mul_(targets['matrix']/expected[0].float().square().mean().sqrt())
        previous=base
        # Raw velocity is retained only to prove the test distinguishes the old rule.
        old_velocity=delta
        actual_base=[p.detach().clone() for p in model.parameters()]
        model.seen.clear()
        module.update(model,optimizer,Tape(),index*4,args,eta,1.,targets,1.)
        assert len(model.seen)==4
        for seen in model.seen:
            for actual,reference in zip(seen,lookahead):
                torch.testing.assert_close(actual,reference,rtol=4e-6,atol=4e-6)
        for p,reference,old in zip(model.parameters(),expected,actual_base):
            torch.testing.assert_close(p,reference,rtol=4e-6,atol=4e-6)
            assert torch.equal(optimizer.state[p]['previous_parameter'],old)
            assert 'velocity' not in optimizer.state[p]
        assert not optimizer.at_lookahead
    assert max_y_gap_from_raw_velocity>1e-3
    before=[p.detach().clone() for p in model.parameters()]
    saved_previous=[optimizer.state[p]['previous_parameter'].clone() for p in model.parameters()]
    try:
        with optimizer.lookahead():
            raise RuntimeError('intentional forward failure')
    except RuntimeError:
        pass
    for p,value,prev in zip(model.parameters(),before,saved_previous):
        assert torch.equal(p,value)
        assert torch.equal(optimizer.state[p]['previous_parameter'],prev)
    # Resume must carry previous base theta, not a raw velocity.
    other=Quadratic(device)
    other.load_state_dict(model.state_dict())
    restored=module.build_optimizer(other,args)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for m,o in ((model,optimizer),(other,restored)):
        module.update(m,o,Tape(),16,args,.013,1.,targets,1.)
    assert all(torch.equal(p,q) for p,q in zip(model.parameters(),other.parameters()))
    return dict(device=device,variable_lr_projected_reference=True,
                all_microbatches_at_parameter_difference_lookahead=True,
                previous_base_state_correct=True,exception_restore_bitwise=True,
                resume_next_step_bitwise=True,distinguishes_raw_velocity=True)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results=[]
    folders=[ROOT.parent/'fixednorm_polyak1_nesterov7_b64_20tpp',ROOT]
    for folder in folders:
        manifest=json.loads((folder/'manifest.json').read_text())
        for job in manifest['jobs']:
            if job['optimizer']!='nesterov':
                continue
            name='difference_'+str(len(results))
            spec=importlib.util.spec_from_file_location(name,folder/job['script'])
            module=importlib.util.module_from_spec(spec)
            sys.modules[name]=module
            spec.loader.exec_module(module)
            row=dict(folder=folder.name,id=job['id'],cpu=check(module,job['momentum'],'cpu'))
            if torch.cuda.is_available():
                row['cuda']=check(module,job['momentum'],'cuda')
            results.append(row)
    report=dict(python=sys.version,torch=torch.__version__,results=results,
                full768_training_tested=False)
    (ROOT/'displacement_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(checked_scripts=len(results),all_passed=True),indent=2))


if __name__=='__main__':
    main()
