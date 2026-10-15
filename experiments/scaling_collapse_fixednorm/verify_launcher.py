"""Validate allocation, outer GPU remapping, stage barriers and failure blocking."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import launch
root=Path(__file__).parent
plan=launch.make_plan([str(i) for i in range(8)],Path('/output'),'train*','val*')
assert [[j['gpu_ids'] for j in stage] for stage in plan['large']]==[[['0','1','2','3']]]
assert [[j['gpu_ids'] for j in stage] for stage in plan['remaining']]==[[['4','5','6','7']],[['4','5'],['6'],['7']]]
for stages in plan.values():
    for stage in stages:
        ids=[i for job in stage for i in job['gpu_ids']]
        assert len(ids)==len(set(ids))
        for j in stage:
            cmd=j['command']; ranks=int(cmd[cmd.index('--expected-world-size')+1]); batch=int(cmd[cmd.index('--device-batch-size')+1])
            assert ranks*batch==128
with tempfile.TemporaryDirectory() as temp:
    out=Path(temp); (out/'console').mkdir()
    marker=out/'completed'
    def job(name,code):return dict(name=name,gpu_ids=['0'],command=[sys.executable,'-c',code])
    first=job('first',f"from pathlib import Path; import time; time.sleep(.2); Path({str(marker)!r}).write_text('done')")
    second=job('second',f"from pathlib import Path; assert Path({str(marker)!r}).read_text()=='done'")
    third=job('third',f"from pathlib import Path; assert Path({str(marker)!r}).exists()")
    assert len(launch.run_lane('remaining',[[first],[second,third]],out))==3
    blocked=out/'must_not_exist'
    fail=job('failure','raise SystemExit(7)')
    forbidden=job('blocked',f"from pathlib import Path; Path({str(blocked)!r}).touch()")
    try:launch.run_lane('failure_case',[[fail],[forbidden]],out)
    except RuntimeError: pass
    else:raise AssertionError('failure was not propagated')
    assert not blocked.exists()
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='7,6,5,4,3,2,1,0'
raw=subprocess.check_output([sys.executable,str(root/'launch.py'),'--train-glob','train*','--val-glob','val*','--output','unused','--dry-run'],env=env,text=True)
assert json.loads(raw)['large'][0][0]['gpu_ids']==['7','6','5','4']
print('PASS: allocation, global B128, outer remapping, stage dependency, parallel stage, failure blocking')
