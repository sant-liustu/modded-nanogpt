# 五尺寸：时间位置、冲击幅度和20对新shard重复

覆盖经典五个尺寸，均继承此前WSD stable checkpoint的fixed-initial-RMS norm control、
Adam状态、矩阵peak ELR0.06、gamma绝对LR0.0018、全局B128/context1024。
所有阶段（包括15k续到16k）从起点开始每步测固定val前缀，默认1,048,576 tokens。

| 模型key | 宽度/深度 | 原默认GPU数 | 原每卡batch |
|---|---|---:|---:|
| w384_d12 | 384/12 | 1 | 128 |
| w768_d6 | 768/6 | 1 | 128 |
| w768_d12 | 768/12 | 2 | 64 |
| w768_d24 | 768/24 | 4 | 32 |
| w1536_d12 | 1536/12 | 4 | 32 |

launcher从实际checkpoint读取rank、microbatch、compile设置和数据glob；无需手填这些参数。
各作业顺序执行，可复用最多4张卡（若原checkpoint可见卡数不同，按其状态要求提供）。
CUDA_VISIBLE_DEVICES会按每个checkpoint保存的可见设备数截取，保持其RNG环境。

## 三类实验

### 1. 不同位置：5000/10000/15000/16000

每个位置运行baseline、elr001、elr003：
- baseline：ELR0.06恒定520步。
- elr001：ELR0.01跑120步，再0.06跑500步。
- elr003：ELR0.03跑40步，再0.06跑500步。

脉冲intrinsic width均1.2，总intrinsic time31.2。完全恢复保存的train loader与预取batch，
不换shard、不重置Adam。每个位置与各自baseline比较，再跨位置比较响应。

### 2. 不同幅度：只在16000

baseline加以下5条脉冲：0.05×24、0.04×30、0.03×40、0.02×60、0.01×120步，
然后全部恢复0.06跑500步。总intrinsic time同为31.2。
all模式下16000的baseline/elr001/elr003与位置实验共用，不重复训练。

### 3. 去波动：每个模型16000做20对

20个不同的新shard，每个shard做baseline520步与elr001的620步，合计40次续训/模型。
跳过checkpoint当前shard及之前全部shard，固定--shard-seed打乱后面的shard列表，
每组从选中shard开头连续读。只shuffle shard，不打乱token；不预检shard长度、不重放历史。
用保存的大小元数据做便宜的完整语料回卷判断；不足20个剩余shard或实际读完会报错。
初次fresh分叉替换旧prefetched batch；同分支中途恢复保持选中shard/游标/下个batch。

每模型12条位置探针+6条幅度探针-3条重复=15条historical，再加40条fresh，
合计55条；五模型共275条续训。若都缺16k，再额外5条15k→16k准备作业。

## Checkpoint配置

复制checkpoints.example.json为自己的配置，替换各模型真实路径。
5000/10000/15000接受此前五尺寸WSD stable脚本对应模型的原始checkpoint。
已有w768/d12的旧shared-stable16000版本继续支持（包括换行哈希兼容）。

其他模型没有16k时，把16000写null或不填，launcher自动先从其15k在0.06下续训1000步，
把生成的共同16k用于所有后续16000分支。不要用原stable脚本的--stop-step16000来代替，
因为原脚本15000后会进入预置decay。本套的stable准备分支明确保持0.06。

新16k可接受本套对应模型脚本生成的shared stable checkpoint；错模型/错调度会拒绝。
同分支中途恢复需保持原--fork-step/experiment/fresh-replicate/shard-seed/fresh-repeats/val预算。
完整模型、Adam、scheduler、norm targets、RNG、loader和prefetch均保存；运行环境仍须与旧checkpoint一致。

## 启动和分批运行

从本目录在云端执行（先替换JSON里的路径）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_suite.py --config checkpoints.json --output outputs/response_all --dry-run
```

确认jobs与输入路径后，删除--dry-run开始。dry-run只读取已有checkpoint元数据，
未来将生成的16k不会被要求提前存在，也不训练。默认覆盖配置中的全部模型及三类实验。
模型检查/源码兼容问题会在元数据阶段报出；训练入口仍保留严格恢复校验。

只跑某个模型/某类：

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_suite.py --config checkpoints.json --models w768_d12 --phase time --output outputs/w768_time --dry-run
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_suite.py --config checkpoints.json --phase prepare --output outputs/prepare16k --dry-run
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_suite.py --config checkpoints.json --phase amplitude --output outputs/amplitude --dry-run
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_suite.py --config checkpoints.json --phase repeats --repeats 20 --shard-seed 0 --output outputs/repeats --dry-run
```

分别跑phase时，把prepare生成的16k路径填回配置，避免重复准备。
历史分支去重只发生在一次all计划内，不会自动跨输出目录复用先前结果。
每次输出根目录必须新建；任意作业失败即停并保留结果。
不要同时运行多条命令占用相同GPU。

## 单独入口

每个模型一个独立train_<modelkey>.py，模型/优化器定义来自已验证版本，不用新的共享trainer。

```bash
torchrun --standalone --nproc_per_node=1 train_w384_d12.py --fork-step 15000 --experiment stable --resume /runs/w384/step015000 --checkpoint-dir outputs/w384_prepare --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin'
torchrun --standalone --nproc_per_node=2 train_w768_d12.py --fork-step 16000 --experiment elr003 --resume /runs/shared/step016000 --checkpoint-dir outputs/w768_003 --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin'
```

GPU/microbatch若与表中默认不同，显式提供--expected-world-size和--device-batch-size以匹配原checkpoint。
--describe-only可不读checkpoint查看调度预算。默认不运行任何verify脚本。

## 分析输出

所有probe_metrics.jsonl记录model_key、fork_step、branch_step、intrinsic_time、模式/replicate、
固定val预算和val_loss，保留脉冲期间完整测量。

完成后自动生成：
- 每模型repeats/aggregate/mean_delta.csv、per_pair_delta.csv和summary.json：20对先做差后平均，sampleSD/SE及描述性逐点正态区间。
- 顶层historical_recovery_delta.csv：所有historical脉冲恢复段的501点delta及delta/(ELR-.06)。

恢复段对齐baseline20+k与pulse(width+k)，完全相同intrinsic time，无需插值。
可独立运行python collect_recovery.py --run-root outputs/response_all。
新shard平均只估计固定起点下未来训练数据差异，不是20次独立预训练，不能消除固定val取样偏差。
不同shard不一定IID；幅度缩放也不等于已经验证叠加性或时不变性。

## 本地开发验证（可选，不是运行前置）

python verify.py
python verify_checkpoint.py

标准库验证覆盖5模型默认配置、125套实际调度/停止块、275任务去重、缺16k依赖、
新shard选择及已知答案的恢复delta；用同一解释器解析参考源片段，避免跨版本AST/token哈希。
CUDA验证覆盖5模型真实meta参数量、280次tiny模型来源迁移/恢复、wrong-model拒绝、
旧w768/d12 shared16k兼容、真实loader取数和20对数据身份、实际LR/gamma及下一步逐位恢复。
这是本地小模型与规划检查；真实用户checkpoint和正式尺寸DDP/compile未实测，未启动正式训练。

## w768/d12再跑40次：只换shard seed

上一批默认--shard-seed0；新入口默认--shard-seed1，仍是20对/40次续训。
同一checkpoint、val、ELR与训练步数不变，只改变剩余shard列表的随机排列。
不排除上一批选过的shard，因此两批可能重叠；单批内部仍取20个不同shard。
若seed1已经使用过，可显式--shard-seed2或其他未使用seed。

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_more40_w768.py --config checkpoints.json --output outputs/w768_repeats_seed1 --dry-run
```

删掉--dry-run开始。配置中的w768/d12应填已有shared16k路径，输出目录换新。
此入口仅给原run_suite.py加默认选项，不更改trainer或checkpoint兼容性。
