"""No production training: fixed config, unchanged code and tiny CUDA checks."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent


def definitions(text):
    return {n.name:ast.dump(n, include_attributes=False) for n in ast.parse(text).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


class ToyTape:
    def batch(self, position, size, device):
        values = (torch.arange(position*8, (position+size)*8+1) % 64).to(device)
        return values[:-1].view(size,8), values[1:].view(size,8)


class HighGradient(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]))
        self.gamma = torch.nn.Parameter(torch.ones(2))

    def forward(self, x, y, return_logits=False):
        return None, 200*self.matrix.sum()+300*self.gamma.sum()


def clipping_check(module, args, device):
    model = HighGradient().to(device)
    optimizer = module.build_optimizer(model,args)
    targets = module.targets_for(model)
    _, before = module.update(model,optimizer,ToyTape(),0,args,args.lr,1.,targets,1.)
    expected_norm = math.sqrt(4*200**2 + 2*300**2)
    assert math.isclose(before, expected_norm, rel_tol=1e-6)
    after = float(torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt())
    assert after <= 50.00001
    expected_gamma = 1 - args.lr*300*50/(expected_norm+1e-6)
    torch.testing.assert_close(model.gamma, torch.full_like(model.gamma,expected_gamma),
                               rtol=2e-6,atol=2e-6)
    # Repeat with microbatch1: clipping must happen once after full accumulation.
    args2 = copy.copy(args)
    args2.microbatch = 1
    other = HighGradient().to(device)
    opt2 = module.build_optimizer(other,args2)
    module.update(other,opt2,ToyTape(),0,args2,args2.lr,1.,module.targets_for(other),1.)
    for x,y in zip(model.parameters(),other.parameters()):
        torch.testing.assert_close(x,y,rtol=2e-6,atol=2e-6)
    return dict(preclip_norm=before,postclip_norm=after,includes_gamma=True,
                clipping_after_accumulation=True)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((ROOT/'manifest.json').read_text())
    sources = ROOT/'tests'
    report = []
    for index, job in enumerate(manifest['jobs']):
        path = ROOT/job['script']
        text = path.read_text()
        ast.parse(text, feature_version=(3,10))
        current = definitions(text)
        template_text = (sources/f"train_{job['optimizer']}.py").read_text()
        template = definitions(template_text)
        assert current.keys() == template.keys()
        assert all(current[k] == template[k] for k in current if k!='parser')
        spec = importlib.util.spec_from_file_location('trial_'+str(index),path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        args = module.parser().parse_args([])
        module.resolve_training_plan(args)
        assert args.effective_lr == job['effective_lr']
        assert args.momentum == job['momentum']
        assert (args.batch,args.microbatch,args.parameter_count,args.target_train_tokens,
                args.updates,args.warmup_updates)==(job['batch'],job['batch'],123590400,2471808000,job['updates'],job['warmup_updates'])
        assert args.output == 'runs_explore8/'+job['id']
        args.lr = args.effective_lr*(1-args.momentum)
        assert math.isclose(args.lr,job['actual_peak_lr'])
        assert math.isclose(module.base_lr(args,0),args.lr/job['warmup_updates'])
        assert module.base_lr(args,job['warmup_updates']-1)==module.base_lr(args,job['updates']-1)==args.lr
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.context,args.batch,args.microbatch=8,4,4
        args.target_train_tokens,args.device=96,device
        clipping = clipping_check(module,args,device)
        torch.manual_seed(0)
        # Small model with width-relative logit scale one, for an optimizer smoke test.
        config=module.GPTConfig(vocab_size=64,n_layer=1,n_head=2,n_embd=16,scale_base_model=16)
        model=module.GPT(config).to(device)
        model.loss_chunk_tokens = 5
        optimizer=module.build_optimizer(model,args)
        targets=module.targets_for(model)
        tape=ToyTape()
        losses=[]
        for step in range(2):
            loss,_=module.update(model,optimizer,tape,step*4,args,args.lr,1.,targets,1.)
            assert math.isfinite(loss)
            losses.append(loss)
        state=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
        reference=module.GPT(config).to(device)
        reference.loss_chunk_tokens = 5
        reference.load_state_dict(state['model'])
        restored=module.build_optimizer(reference,args)
        restored.load_state_dict(state['optimizer'])
        for candidate,opt in ((model,optimizer),(reference,restored)):
            loss,_=module.update(candidate,opt,tape,8,args,args.lr,1.,targets,1.)
            assert math.isfinite(loss)
        assert all(torch.equal(p,q) for p,q in zip(model.parameters(),reference.parameters()))
        diagnostics=module.diagnostics(model,targets,1.,args.lr,args.lr)
        error=max(x.get('relative_error',0) for x in diagnostics.values())
        assert error<3e-6
        report.append(dict(id=job['id'],device=device,peak_actual_lr=args.lr,
                           clipping=clipping,
                           tiny_three_steps_finite=True,optimizer_restore_next_step_bitwise=True,
                           norm_max_relative_error=error,matches_clipped_template_except_parser=True))
    result=subprocess.run([sys.executable,str(ROOT/'launch.py'),'--train','/data/train_*.bin',
        '--val','/data/val_*.bin','--output-root','/runs/search8','--dry-run'],
        capture_output=True,text=True,check=True)
    lines=result.stdout.splitlines()
    assert len(lines)==8
    for index,line in enumerate(lines):
        assert line.startswith(f'CUDA_VISIBLE_DEVICES={index%8} ')
        assert manifest['jobs'][index]['script'] in line
    output=dict(python=sys.version,torch=torch.__version__,jobs=report,
                eight_trials_on_eight_gpu_dry_run=True,
                default768_batch32_training_or_memory_tested=False,
                cloud_linux_python310_training_tested=False)
    (ROOT/'verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
