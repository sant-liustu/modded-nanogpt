# w768/d12：140对、280次两shard混合重复

从同一个shared-stable16000完整checkpoint出发，新增140对（280次）续训：
- baseline：ELR0.06恒定270步。
- elr001：ELR0.01跑120步，再恢复0.06跑250步。

保留模型、Adam moments、scheduler、固定初始矩阵RMS、gamma LR0.0018及模型RNG。
默认2卡/每卡B64/全局B128；launcher从真实checkpoint读取运行设置。
所有分支起点和每次更新后都测同一个val前缀，默认1,048,576 tokens。

## 两份shard怎样提供140条不同轨迹

每对用一个新的数据seed，默认2000–2139。默认每对随机选两份在16k checkpoint当前
shard之后的训练shard；可显式固定两份shard，让所有组共享这个池。

1. 把两份uint16 shard分别读入内存，不把两份文件边缘硬接成训练句子。
2. 每份按完整context切成1024-token窗口，label为其后移1 token；末尾不足窗口的部分舍弃。
3. 用私有seed打乱两份文件的context ID排列，不放回地取context来组batch。
4. 每条context内部的token顺序不变；同组的两个arm使用相同两份文件和相同排列。
5. 所有rank共享全局排列，各自读取本次global microbatch中不重叠的行。

因此不仅batch顺序变化，batch内的context组合也变化。无需140个新shard。
不同组可以复用语料，不能称为140份独立新语料。只要求这些shard未被共同16k起点
训练过；它们可能在旧的其他分叉运行中使用过，新的混合排列仍提供不同轨迹。
这是与之前“单shard顺序读”不同的采样协议，本批单独汇总，不自动混入旧结果。

两个100M-token shard约占400MB主机RAM/每个rank（uint16），另有少量排列索引和gather缓冲。
每个进程只读入自己的两份池一次，训练期间从RAM取窗口，不按每步重新读整份shard。

## 运行

从本目录在云端、保持原checkpoint的PyTorch/CUDA环境执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_mix2.py --resume /runs/shared/step016000 --output outputs/w768_mix140_250 --dry-run
```

去掉--dry-run开始140对/280次续训，顺序复用两张GPU。无需手填训练/验证glob和microbatch，
launcher直接读取checkpoint原配置。default --pairs140 --seed-start2000。
可用--pairs修改对数，--seed-start选新的数据seed范围。

固定两份未训练shard（index要大于checkpoint当前shard）：

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_mix2.py --resume /runs/shared/step016000 --output outputs/w768_fixedpool140_250 --pool-shards 23 24 --dry-run
```

默认仅保存loss、数据计划和汇总，不给280个短分支保存大批模型文件。
需要完整分支checkpoint时加--save-checkpoints（pulse结束及最终步）；
或单独运行trainer并用--save-at/--stop-after-save保存某个中途状态。

## 输入兼容与恢复

支持旧w768/d12 shared16k的两种换行哈希版本，以及五模型套件train_w768_d12.py生成
的shared-stable16000版本。16k须是stable0.06，不接受原脚本已decay的16k。
继续保留model/config、数据文件、rank、microbatch、Torch/CUDA等原恢复检查。

首次从历史16k迁移：只替换后续训练数据及旧prefetched batch，不重置模型/Adam。
新mixed checkpoint保存两份文件index、完整context排列、排列digest、游标和下一batch。
同分支恢复时保持experiment/mix-seed/mix-shards/val预算及原运行设置，不重新shuffle。
默认只输出指标的运行没有中途模型checkpoint；失败的短分支需要从原16k重跑。

单独一条（新输出目录）：

```bash
torchrun --standalone --nproc_per_node=2 train_w768_mix2.py --experiment elr001 --mix-seed 2000 --mix-shards 23 24 --resume /runs/shared/step016000 --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/pulse2000 --log-root outputs/logs --run-name pulse2000
```

旧config中val_loss_every=0只是为历史恢复保留的字段，本脚本实际固定每步评估。

## 平均与输出

manifest.json记录140个seed、每对两份shard和280条命令。
logs/<seed_arm>/<UUID>/mix_data_metadata.json记录池和排列指纹；probe_metrics.jsonl
记录seed、shards、physical step、intrinsic time、固定val预算与val_loss。

完成后自动运行：

```bash
python aggregate_mix_pairs.py --run-root outputs/w768_mix140_250
```

先逐对计算pulse-minus-baseline delta，再对140个delta取均值、sampleSD、SE。
共同tau网格0:.06:16.2，271点；baseline第i步对应pulse第6i步(i<=20)，之后第i+100步。
精确整数步对齐，不插值。输出aggregate/per_pair_delta.csv、mean_delta.csv、summary.json。
拒绝缺失/冲突/非有限点、组内不同排列、跨组重复完整轨迹、初始val不一致等数据错误。

同一val集上的共同取样误差不会因重复140次消失。区间为跨数据轨迹的描述性逐点正态近似，
不是140次独立预训练的不确定性，也不是对LTI/信号噪声分解的证明。

## 本地检查（不自动在训练前执行）

python verify.py
python verify_checkpoint.py

真实Python3.10/3.12：140种相同seed可重现且跨seed不同的context排列，280作业、
原模型/optimizer定义、实际调度与两种保存策略的停止边界、已知答案统计。
Tiny CUDA：实际context/label连续性、最后窗口不跨shard、rank划分、池耗尽拒绝、
140种配对轨迹和280次完整迁移/磁盘恢复，下一步loss/参数/Adam/scheduler/batch/order/cursor
逐位一致，初始val和全局RNG不变，三种历史shared16k来源及真实mmap规划。
未读取用户真实checkpoint，未在云端Linux3.10.16或正式尺寸DDP/compile上运行。

本版恢复段为250步，基线270步、脉冲370步，总tau16.2。使用新输出目录，勿与旧版本日志混用；旧版分支checkpoint需用生成它的原代码恢复。
