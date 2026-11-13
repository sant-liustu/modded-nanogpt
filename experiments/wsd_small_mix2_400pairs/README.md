# 两个小模型：每个400对混合上下文脉冲重复

覆盖w384/d12（40,563,072参数）和w768/d6（81,112,320参数）。
每个模型400对、800次续训；两模型合计800对、1600次。

| 分支 | 调度 | 新增更新 | 最终step | 总intrinsic time |
|---|---|---:|---:|---:|
| baseline | 0.06恒定 | 270 | 16270 | 16.2 |
| elr001 | 0.01跑120步，再0.06跑250步 | 370 | 16370 | 16.2 |

每组都从该模型同一个shared-stable16000 checkpoint恢复，保留Adam/scheduler、
模型RNG、初始矩阵RMS norm control及gamma绝对LR0.0018。
每步测固定val前缀，默认1,048,576 tokens；模型之间单独汇总。

## 数据重复方式

数据seed默认4000–4399。每对从checkpoint当前shard之后随机选两份shard，读入两份
uint16数组；按完整1024-token上下文切分，把context ID顺序打乱后组batch。
上下文内部token顺序保持不变，next-token标签不会跨shard边缘。
同对两条分支使用相同shard和排列；各rank读同一全局排列的不重叠行。
不同组可复用语料，但有不同数据排列；不是400份独立新语料。
只shuffle完整上下文，不shuffle单个token；不重放历史训练，不逐份预检shard长度。

沿用最新w768混合协议：初次历史迁移替换旧prefetched batch，同分支恢复保存的
完整排列、cursor和next batch。默认只保存指标和数据计划；--save-checkpoints可开
完整分支保存，或单独trainer --save-at。默认不为800条短任务保存大量模型权重。

## 已有shared16k：单模型运行

默认原配置两模型各1卡/每卡B128/全局B128。launcher从checkpoint读取实际卡数、
microbatch、compile设置和数据glob，原配置若被调整过则以checkpoint为准。

```bash
CUDA_VISIBLE_DEVICES=0 python run_small_mix2.py --model w384_d12 --resume /runs/w384_shared/step016000 --output outputs/w384_mix400 --dry-run
CUDA_VISIBLE_DEVICES=0 python run_small_mix2.py --model w768_d6 --resume /runs/w768_d6_shared/step016000 --output outputs/w768d6_mix400 --dry-run
```

去掉--dry-run开始。--pairs默认400，--seed-start默认4000。
每次输出目录必须新建。dry-run仅读已有checkpoint元数据并列出命令，不训练。
若固定同两份未训练shard重复，可加 --pool-shards 23 24。

两张空闲GPU且原checkpoint均为单卡时，可以在两个终端分别用CUDA_VISIBLE_DEVICES=0
和CUDA_VISIBLE_DEVICES=1执行上述两个模型；否则不要让两条命令占用同一GPU。

## 一次规划两个模型

将checkpoints.example.json复制为checkpoints.json并填入真实shared16k路径：

```bash
CUDA_VISIBLE_DEVICES=0 python run_small_mix2.py --config checkpoints.json --output outputs/small_mix400 --dry-run
```

默认依次运行两个模型，共1600条，不自动并行。相同seed也便于两个模型采用可比较的
数据随机规则；两模型的val loss和delta始终各自计算，不能把两者当成800个同模型样本。

## 只有15k时，准备共同16k

已包含此前五模型套件对应脚本的字节相同副本，因此生成16k的来源哈希与旧套件一致。
输入15k须为原五尺寸WSD stable checkpoint。准备阶段保持矩阵ELR0.06且每步测val：

```bash
torchrun --standalone --nproc_per_node=1 prepare_w384_d12_16000.py --fork-step 15000 --experiment stable --resume /runs/w384/step015000 --checkpoint-dir outputs/w384_shared --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin'
torchrun --standalone --nproc_per_node=1 prepare_w768_d6_16000.py --fork-step 15000 --experiment stable --resume /runs/w768d6/step015000 --checkpoint-dir outputs/w768d6_shared --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin'
```

保持数据glob、runtime/rank/microbatch与原checkpoint一致；如非1卡B128，显式传
--expected-world-size和--device-batch-size。输出各自step016000再交给mixed入口。
不要直接把原stable脚本延长到16k，它会在15k后进入旧decay。

## 兼容和结果

支持此前五模型response suite或上述准备脚本生成的对应模型shared16k。
不接受w768/d12权重、错模型/错调度、已decay的16k或其他未记录来源。
保留原PyTorch/CUDA、数据路径/mtime、rank/microbatch、配置、源码校验。
云端是Python3.10.16，本地是3.12.3；跨版本自检采用同解释器解析参考源片段，
不使用跨版本ast.dump/tokenizer哈希。未自动放宽原checkpoint运行环境检查。

每个模型输出在OUTPUT/<modelkey>/：manifest.json、console/、logs/和aggregate/。
完成后自动逐对做pulse-minus-baseline，再输出400对的mean/SD/SE及描述性逐点正态区间。
单独重算：

```bash
python aggregate_mix_pairs.py --run-root outputs/small_mix400/w384_d12
python aggregate_mix_pairs.py --run-root outputs/small_mix400/w768_d6
```

共同tau网格0:.06:16.2共271点；恢复段baseline20+k与pulse120+k，精确对齐。
固定val取样偏差不会因为重复400次消失；区间只描述给定checkpoint和数据协议的变异。
这批mixed数据协议与更早单shard顺序实验分开分析，不自动混入旧结果。

同分支续训需用原model/experiment/mix-seed/mix-shards/val预算，恢复保存的排列/游标，
不重新shuffle。若未启用完整checkpoint，则失败短分支需从共同16k重跑。

## 开发验证（独立可选，不自动运行）

python verify.py
python verify_checkpoint.py

真实Python3.10/3.12标准库检查：400排列/模型、800任务/模型、actual250/270/370
调度与停止块、模型来源绑定、准备脚本一致性、已知答案的独立模型统计。
CUDA检查：两种完整参数形状、每模型400实际context/rank/label检查、代表性16次完整
磁盘迁移/恢复、错模型/seed拒绝、实际mmap1600任务规划。另各做一次真实小模型形状
B1/T16 eager BF16前向/反向/Adam/范数投影，检查loss有限和矩阵RMS恢复。
这些不代表正式B128/T1024、DDP/compile或真实用户checkpoint已经运行。
未启动正式训练。
