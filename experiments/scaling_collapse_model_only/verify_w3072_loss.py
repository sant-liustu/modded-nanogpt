"""CPU equivalence check without importing the CUDA trainer."""
import ast
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
source=Path(__file__).with_name('train_fixednorm_w3072.py').read_text()
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='chunked_linear_ce')
ns=dict(torch=torch,F=F,checkpoint=checkpoint)
exec(compile(ast.Module(body=[node],type_ignores=[]),'loss_helper','exec'),ns)
for chunk in [1,4,32]:
    torch.manual_seed(3)
    w=torch.randn(19,8,requires_grad=True)
    ids=torch.tensor([[1,3,5,7,9],[2,4,6,8,10]])
    targets=torch.tensor([[2,-1,6,8,10],[3,5,7,-1,11]])
    h=F.embedding(ids,w)
    ref=F.cross_entropy((F.linear(h,w)/4).float().reshape(-1,19),targets.reshape(-1),ignore_index=-1)
    grad_ref=torch.autograd.grad(ref,w)[0]
    h=F.embedding(ids,w)
    got=ns['chunked_linear_ce'](h,w,targets,4,chunk)
    grad_got=torch.autograd.grad(got,w)[0]
    torch.testing.assert_close(got,ref,rtol=2e-6,atol=2e-6)
    torch.testing.assert_close(grad_got,grad_ref,rtol=2e-5,atol=2e-6)
    with torch.no_grad():
        val=ns['chunked_linear_ce'](F.embedding(ids,w),w,targets,4,chunk)
    torch.testing.assert_close(val,ref)
print('PASS: loss, tied embedding/head gradients, ignored labels, chunk tails, width scaling, no-grad evaluation')
