"""Stdlib-only checks for the400-pair mixed-context plan."""
import ast, hashlib, json, random, runpy, sys
from pathlib import Path
from unittest.mock import patch
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from run_mix2 import make_plan
info=dict(ranks=2,device_batch_size=64,context=1024,batch_size=128,compile_model=1,input_bin='train*.bin',input_val_bin='val*.bin',train_shard=20,train_files=[str(i) for i in range(100)],train_sizes=[1024+2*100000000]*100)
old=make_plan(info,'/mock/shared16000','/mock/old')
new=make_plan(info,'/mock/shared16000','/mock/new',pairs=260,seed_start=2140)
assert len(old['jobs'])==280 and len(new['jobs'])==520
assert old['replicates']+new['replicates']==list(range(2000,2400))
for plan in (old,new):
    for a,b in zip(plan['jobs'][::2],plan['jobs'][1::2]):
        assert a['replicate_id']==b['replicate_id'] and a['mix_shards']==b['mix_shards']
        assert a['experiment']=='baseline' and b['experiment']=='elr001'
        assert min(a['mix_shards'])>20
ns={'random':random};tree=ast.parse((root/'train_w768_mix2.py').read_text(encoding='utf-8'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_context_order')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<order>','exec'),ns)
digests={hashlib.sha256(json.dumps(ns['make_context_order']([1001,1201],seed)).encode()).hexdigest() for seed in range(2000,2400)}
assert len(digests)==400
with patch('run_mix2.main') as main,patch.object(sys,'argv',['run_topup400.py','--dry-run']):
    runpy.run_path(str(root/'run_topup400.py'),run_name='__main__')
    main.assert_called_once_with(default_pairs=260,default_seed_start=2140)
print('PASS:140+260 pairs,800 runs,400 unique context orders;top-up defaults and pairing')
