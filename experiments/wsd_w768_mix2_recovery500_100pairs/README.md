# w768/d12：100对，恢复500步

脉冲保持ELR0.01跑120步（intrinsic width1.2），恢复0.06后再跑500步。配对基线保持0.06跑520步。两条分支总intrinsic time均为31.2，分别停止于16520和16620步。

本批100对、200次续训，seed6000–6099。另一个恢复时长使用不重叠seed，两批分开汇总。研究问题是延长观测窗口后的delta尾部；固定val误差不会因重复自动消失。

从同一个w768/d12 shared-stable16000完整checkpoint重新分叉；保留norm control、Adam、scheduler、模型RNG、gamma LR0.0018。每步测固定val，默认1048576 tokens。每对从16k未训练后缀中随机选两份shard，shuffle完整1024-token上下文；组内池/排列相同，不打乱内部token。不同组可复用语料。

## 运行

在本目录，保持原checkpoint的PyTorch/CUDA环境：

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_mix2.py --resume /runs/shared/step016000 --output outputs/w768_mix100_recovery500 --dry-run
```

检查后去掉--dry-run。默认100对，无需传--pairs。原配置2卡/每卡B64；launcher从实际checkpoint读取卡数、microbatch、compile和数据路径。可用--pool-shards固定两份未训练的shard。默认只保存指标；--save-checkpoints可选完整分支checkpoint。没有分支checkpoint的失败短任务从共同16k重跑。

完成后自动逐对计算pulse-minus-baseline，在0:.06:31.2的521点精确网格汇总mean/SD/SE与描述性逐点区间。恢复段baseline20+k对应pulse120+k，不插值。不混入250步或另一恢复时长结果。本版只接受对应recovery500的manifest。

支持原三种已知w768/d12 shared-stable16k来源；保留源码/配置/运行环境检查。旧250步分支checkpoint不能用本版本续训。同版本恢复保留数据排列、cursor和next batch。

## 审阅与验证

审阅120步脉冲不变、500步恢复、520/620停止边界和521点对齐。复制独立训练/启动/聚合版本，没有引入新抽象。verify.py检查实际调度/停止代码、100种排列、200任务和已知答案统计；verify_checkpoint.py是可选tiny CUDA恢复检查，不自动在训练前执行。真实云端checkpoint及正式DDP/compile未运行，未启动正式云端训练。
