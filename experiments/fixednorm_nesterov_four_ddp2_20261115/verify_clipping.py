import importlib.util,json,math,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parent
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.matrix=torch.nn.Parameter(torch.tensor([[1.,2.],[3.,4.]]));self.gamma=torch.nn.Parameter(torch.ones(2))
    def forward(self,x,y,return_logits=False):return None,200*self.matrix.sum()+300*self.gamma.sum()
class Tape:
    def batch(self,position,size,device):return torch.zeros(size,1),torch.zeros(size,1)
rows=[]
for i,j in enumerate(json.loads((ROOT/'manifest.json').read_text())['jobs']):
    spec=importlib.util.spec_from_file_location('clip_'+str(i),ROOT/j['script']);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    a=SimpleNamespace(lr=.01,momentum=.98,arm='fixed',context=1,batch=2,microbatch=1,target_train_tokens=2,device='cpu')
    model=Model();o=m.build_optimizer(model,a);targets=m.targets_for(model)
    loss,norm=m.update(model,o,Tape(),0,a,.01,1.,targets,1.)
    expected=math.sqrt(4*200**2+2*300**2);assert math.isclose(norm,expected,rel_tol=1e-6)
    actual=float(torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt())
    if not j['grad_clip']:
        assert actual>580
        torch.testing.assert_close(model.gamma,torch.full_like(model.gamma,-2.))
    else:assert actual<=50.00001
    rows.append(dict(id=j['id'],preclip_norm=norm,postclip_norm=actual,clip_enabled=bool(j['grad_clip'])))
(ROOT/'clipping_verification.json').write_text(json.dumps(rows,indent=2));print('No-clip and clip50 high-gradient behavior passed for all four scripts.')
