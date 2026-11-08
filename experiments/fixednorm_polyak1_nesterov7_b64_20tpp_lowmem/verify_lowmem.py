"""Full-logit reference versus chunked checkpointed loss and head peak memory."""
import contextlib
import copy
import gc
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def relative_error(actual, expected):
    numerator = sum((x.float()-y.float()).square().sum() for x,y in zip(actual,expected))
    denominator = sum(y.float().square().sum() for y in expected)
    return float((numerator/denominator.clamp_min(1e-30)).sqrt())


def full_model_equivalence(module, device, mixed):
    torch.manual_seed(123)
    config = module.GPTConfig(vocab_size=256,n_layer=2,n_head=2,n_embd=16,scale_base_model=16)
    reference = module.GPT(config).to(device)
    candidate = copy.deepcopy(reference)
    reference.loss_chunk_tokens = 0
    candidate.loss_chunk_tokens = 13
    # Batch64 remains a single Transformer forward; mask creates all-ignored chunks.
    x = torch.arange(64*16,device=device).reshape(64,16)%256
    y = (x+1)%256
    y[-2:,:] = -1
    gradients=[]
    losses=[]
    for model in (reference,candidate):
        amp = torch.autocast('cuda',dtype=torch.bfloat16) if mixed else contextlib.nullcontext()
        with amp:
            logits,loss = model(x,y,return_logits=False)
        assert logits is None
        losses.append(float(loss.detach()))
        loss.backward()
        gradients.append([p.grad.detach().clone() for p in model.parameters()])
    error=relative_error(gradients[1],gradients[0])
    assert abs(losses[1]-losses[0]) < (2e-3 if mixed else 2e-6)
    assert error < (.02 if mixed else 2e-5)
    reference.eval();candidate.eval()
    with torch.no_grad():
        _,a=reference(x,y,return_logits=False)
        _,b=candidate(x,y,return_logits=False)
    assert abs(float(a-b)) < 2e-6
    # Returning logits and last-token inference retain the original code path.
    with torch.no_grad():
        a, _ = reference(x,None)
        b, _ = candidate(x,None)
    assert torch.equal(a,b)
    return dict(device=device,bfloat16_autocast=mixed,batch=64,context=16,
                reference_loss=losses[0],chunked_loss=losses[1],gradient_relative_l2_error=error,
                masked_validation=True,inference_equal=True)


def head_memory(module):
    # Actual vocabulary, smaller hidden width/token count to fit the local GPU.
    config=module.GPTConfig(vocab_size=50304,n_layer=1,n_head=2,n_embd=128,scale_base_model=128)
    model=module.GPT(config).cuda()
    hidden=torch.randn(1,2048,128,device='cuda',requires_grad=True)
    targets=torch.arange(2048,device='cuda').reshape(1,-1)%50304
    measurements={}
    for chunk in (0,128):
        model.zero_grad(set_to_none=True);hidden.grad=None
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        baseline=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if chunk:
                model.loss_chunk_tokens=chunk
                loss=model.chunked_lm_loss(hidden,targets)
            else:
                logits=model.lm_head(hidden)/model.width_multiplier
                loss=torch.nn.functional.cross_entropy(logits.float().view(-1,50304),targets.view(-1))
                del logits
        loss.backward();torch.cuda.synchronize()
        measurements[str(chunk)]=dict(loss=float(loss.detach()),
            incremental_peak_bytes=torch.cuda.max_memory_allocated()-baseline)
        del loss
    assert measurements['128']['incremental_peak_bytes'] < measurements['0']['incremental_peak_bytes']*.5
    return dict(tokens=2048,vocab=50304,width=128,head_only=True,measurements=measurements,
                full_w768_batch64_peak_memory_tested=False)


def main():
    torch.set_num_threads(1)
    module=load('memory_variant',ROOT/'train_nesterov_elr0p96_beta0p9.py')
    report=dict(equivalence=[full_model_equivalence(module,'cpu',False)])
    if torch.cuda.is_available():
        report['equivalence'].append(full_model_equivalence(module,'cuda',True))
        report['head_memory']=head_memory(module)
    (ROOT/'lowmem_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
