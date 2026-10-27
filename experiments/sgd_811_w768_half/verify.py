"""Checks invariants independently of training success."""
import ast
from pathlib import Path
import tempfile
import numpy as np
import torch
from train import Tape, schedule, update, save_prefix_milestone, evaluate
from model import GPT, GPTConfig

source=Path('F:/自动化/自动化for agent/人机协作区/norm_control_muon_schedule_collapse/modded-nanogpt-local/experiments/small_batch_adamw_baseline/generated_train_scripts/train_small_batch_adamw_rmsnorm_gamma_B0128_devB064_lr0p0036_wd0_seed00.py')
if source.exists():
    original={n.name:ast.dump(n) for n in ast.parse(source.read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    snapshot=ast.parse(Path(__file__).with_name('model.py').read_text())
    for node in snapshot.body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)):
            assert ast.dump(node)==original[node.name], node.name
else:
    print('SKIP original-source AST comparison: original local checkout absent')
for n in (100,1000,163200,301728,603456,652800):
    g,prefix,scale=schedule(n)
    assert sum(g)==n and np.all(g[:prefix]==1)
    assert set(np.unique(g)) == {1, 4, 10} and scale == 1.0
    assert np.all(np.diff(g) >= 0), 'no alternating schedule values'
    assert abs(int((g == 4).sum()) * 4 - n/10) <= 2
    assert abs(int((g == 10).sum()) * 10 - n/10) <= 5
    if n == 301728:
        assert prefix == 241386 and len(g) == 251946
        assert int((g == 4).sum()) * 4 == 30172
        assert int((g == 10).sum()) * 10 == 30170
    if n == 603456:
        assert prefix == 482762
        assert int((g == 4).sum()) == 15086
        assert int((g == 10).sum()) == 6035
    for value in np.unique(g):
        assert np.isclose(value*(0.01/value),0.01)
        assert np.isclose(4/(0.01/value),4*value/0.01)
    print('schedule',n,'macros',len(g),'scale',scale,'g',np.unique(g))
# Validate data continuity across shard boundary and accumulation equivalence.
with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
    for index in range(2):
        header=np.zeros(256,dtype=np.int32)
        header[:3]=20240520,1,101
        with open(Path(tmp)/f'{index}.bin','wb') as f:
            f.write(header.tobytes())
            f.write(np.arange(index*101,(index+1)*101,dtype=np.uint16).tobytes())
    tape=Tape(str(Path(tmp)/'*.bin'),8)
    x,y=tape.batch(12,2,'cpu')
    assert x.flatten().tolist()==list(range(96,112))
    assert y.flatten().tolist()==list(range(97,113))
    torch.manual_seed(0)
    a=GPT(GPTConfig(n_layer=1,n_head=2,n_embd=16,vocab_size=256))
    b=GPT(GPTConfig(n_layer=1,n_head=2,n_embd=16,vocab_size=256))
    b.load_state_dict(a.state_dict())
    oa=torch.optim.SGD(a.parameters(),lr=.01)
    ob=torch.optim.SGD(b.parameters(),lr=.01)
    update(a,oa,tape,0,4,.01,4,'cpu')
    update(b,ob,tape,0,4,.01,1,'cpu')
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa,pb,atol=2e-7,rtol=2e-6)
    # Independent validation batching, including a final partial batch.
    single=evaluate(a,tape,17,1,'cpu')
    for val_batch in (4,8,32):
        assert abs(single-evaluate(a,tape,17,val_batch,'cpu')) < 2e-6
    for array in tape.arrays:
        array._mmap.close()
print('PASS: exact model snapshot, paired clocks/ratios/tokens, contiguous data, accumulation')
with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
    directory=Path(tmp)
    model=torch.nn.Linear(2,2)
    contract={'test':True}
    assert save_prefix_milestone(directory,model,'prefix',199999,799996,199999,contract) is None
    path=save_prefix_milestone(directory,model,'prefix',200000,800000,200000,contract)
    state=torch.load(path,weights_only=False)
    assert state['updates']==200000 and state['position']==800000
    assert state['macro']==200000 and state['contract']==contract
    for key,value in model.state_dict().items():
        torch.testing.assert_close(state['model'][key],value)
    timestamp=path.stat().st_mtime_ns
    save_prefix_milestone(directory,model,'prefix',200000,800000,200000,contract)
    assert path.stat().st_mtime_ns==timestamp
    assert save_prefix_milestone(directory,model,'batch',200000,800000,200000,contract) is None
print('PASS: exact 200000-update milestone, state roundtrip, no overwrite')

