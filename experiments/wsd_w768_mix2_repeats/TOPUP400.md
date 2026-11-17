# w768/d12 补到400对

目的：在相同16k checkpoint、固定val和混合上下文协议下增加数据顺序重复，降低delta均值的抽样波动；不保证固定val误差消失。

原批140对，seed2000–2139。本次新增260对（520次续训），seed2140–2399；两批共400对（800次）。更早单shard协议不计入。

baseline维持0.06跑270步；pulse为0.01跑120步，再0.06跑250步。两份未训练后缀shard的完整1024-token上下文混合，组内配对相同。保留norm control、Adam和每步固定val（默认1048576 tokens）。仅增加重复数，不增加训练抽象或改训练代码。有限次数的均值仍可能波动；这不是独立预训练重复。

复用同一个shared-stable16000 checkpoint和之前的val预算、pool选项。如果之前固定了--pool-shards，本次保持一致。

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_topup400.py --resume /runs/shared/step016000 --output outputs/w768_mix260_topup400 --dry-run
```

检查后去掉--dry-run开始；默认读取checkpoint的实际卡数和microbatch。与原批相同，默认只保存指标。补跑目录必须全新。此入口固定补260对且排除旧seed，不能传--pairs或--seed-start。

若从零重跑完整400对，用原入口：

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_mix2.py --resume /runs/shared/step016000 --output outputs/w768_mix400 --pairs 400 --seed-start 2000 --dry-run
```

每个输出目录自动独立汇总。两批指标应按全部400条逐对delta重新统计；不要把两批SE直接平均。旧批只有在相同checkpoint/val预算/250步恢复和混合数据协议下才能合并。本次不自动读取或更改旧结果。

审阅：新增seed范围、520条命令及组内数据配对。验证：Python3.10/3.12实际生成两批计划，确认800条作业、400个不重复seed和context排列；沿用原训练验证，未读取真实云端checkpoint或启动训练。
