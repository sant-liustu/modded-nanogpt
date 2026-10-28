"""Check accumulate-before-square against an explicit full objective."""
import torch
from fisher_diagnostics import probe, parameter_rms, should_measure
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Module()
        self.block.mlp = torch.nn.Linear(3, 5)
    def forward(self, x, y, return_logits=True):
        return self.block.mlp(x), None

torch.manual_seed(73)
m=Tiny()
x=torch.randn(7,4,3)
y=torch.zeros(7,4,dtype=torch.long)
y[0,:]=-1
y[4,1:]=-1
initial=parameter_rms(m)
for p in m.parameters(): p.grad=torch.randn_like(p)
saved=[p.grad.clone() for p in m.parameters()]
rng=torch.get_rng_state().clone()
out=probe(m,x,y,1,97,initial,micro_batch_size=2)
assert torch.equal(rng,torch.get_rng_state()) and m.training
for p,g in zip(m.parameters(),saved): assert torch.equal(p.grad,g)
# Same labels and inputs, but hold all graphs and differentiate the combined loss.
gen=torch.Generator().manual_seed(97)
loss=0
N=int((y!=-1).sum())
individual=[]
for i in range(0,7,2):
 z=m(x[i:i+2],y[i:i+2])[0][y[i:i+2]!=-1]
 sampled=torch.multinomial(z.detach().softmax(-1),1,generator=gen).squeeze(-1)
 part=torch.nn.functional.cross_entropy(z,sampled,reduction='sum')/N
 individual.append(torch.autograd.grad(part,list(m.parameters()),retain_graph=True))
 loss=loss+part
grads=torch.autograd.grad(loss,list(m.parameters()))
for (name,p),g in zip(m.named_parameters(),grads):
 expected=N*g.double().square().sum().item()
 assert abs(out['parameters'][name]['trace']-expected)<2e-7, name
# Ensure the fixture distinguishes sum-of-squares from square-of-sum.
wrong=N*sum(gs[0].double().square().sum().item() for gs in individual)
assert abs(wrong-out['parameters']['block.mlp.weight']['trace'])>1e-4
assert [s for s in range(3501) if should_measure(s,0,3500,1000)]==[0,100,1000,2000,3000,3500]
assert not any(should_measure(s,0,3500,0) for s in range(3501))
assert out['valid_tokens']==N and out['groups']['FFN']['mc_standard_error'] is None
print('PASS: masked unequal micro-batches; accumulated gradient matches full objective; training grads/RNG/mode preserved; schedule correct')

# Each repeated draw must match a separate estimate using its recorded seed.
repeated=probe(m,x,y,2,97,initial,micro_batch_size=2)
second=probe(m,x,y,1,98,initial,micro_batch_size=2)
assert repeated['draw_seeds']==[97,98]
for section in ('parameters','groups'):
 for name,v in repeated[section].items():
  expected=[out[section][name]['mean_fisher'],second[section][name]['mean_fisher']]
  assert v['mean_fisher_draws']==expected, name
  assert abs(v['mean_fisher']-sum(expected)/2)<1e-15
  assert abs(v['mc_standard_error']-abs(expected[0]-expected[1])/2)<1e-15
# Group uncertainty is calculated after each draw's parameter aggregation.
for draw in range(2):
 ps=list(repeated['parameters'].values())
 expected=sum(p['numel']*p['mean_fisher_draws'][draw] for p in ps)/sum(p['numel'] for p in ps)
 assert abs(repeated['groups']['FFN']['mean_fisher_draws'][draw]-expected)<1e-15
assert torch.equal(rng,torch.get_rng_state()) and m.training
for p,g in zip(m.parameters(),saved): assert torch.equal(p.grad,g)
print('PASS: independent seeds, buffers reset between draws, per-draw values, mean and SE, group aggregation, and isolation')
