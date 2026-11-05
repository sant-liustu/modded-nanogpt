# 16k checkpoint：剩余shard shuffle，20组配对平均

简单流程：读取原16k checkpoint的train_loader.current_shard，整份跳过它及之前的shard。
将后面的shard列表用固定seed打乱，取20个不同shard；每组从其中一个shard开头顺序训练。
只shuffle shard列表，不打乱token，也不打乱shard内部的块。

默认20组，每组两条分支，共40次续训，复用同两张GPU：
- baseline：0.06跑520步。
- elr001：0.01跑120步，再恢复0.06跑500步。

组内使用同一个新shard；不同组使用不同shard。保留16k模型、Adam、gamma及norm control。
每步测同一val前缀，默认1,048,576 tokens；最后先做逐组delta，再输出20组均值/SD/SE。

## 运行

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_repeats.py --resume /runs/shared/step016000 --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --output outputs/fresh_shard20 --shard-seed 0
```

默认--repeats20，--shard-seed0；相同seed得到相同shard列表。加--dry-run只看选中的shard
和40条命令，不训练。保持原checkpoint的数据glob，脚本在原列表内选择后面的shard。
也可用--fresh-shard-indices显式选择；这样按给定列表使用，不再shuffle它。

不会自动运行verify.py或verify_checkpoint.py，不重放历史训练步，不另行逐文件扫描，
也不预检每份新shard长度。默认按这些shard足够长使用。
只用checkpoint已保存的大小元数据做一次简单判断：如果已遍历完一轮语料，就拒绝
把当前shard之后的内容当作“未训练”。当前shard包含已用一部分的数据，因此始终整份跳过。
保留原checkpoint基本恢复校验；实际读到shard末尾仍报错，不偷偷回卷到旧数据。

manifest.json记录shard seed、实际选中的20个index、40条命令；可以直接核对。
全部结束自动汇总。下载结果后也能运行：

```bash
python aggregate_pairs.py --run-root outputs/fresh_shard20
```

输出aggregate/per_pair_delta.csv、mean_delta.csv、summary.json。
共同tau网格0:.06:31.2，共521点，按整数步精确对齐，不插值。
基线第i步对应pulse第6i步(i<=20)，之后对应pulse第i+100步。
输出跨shard的sample SD、SE和描述性逐点mean±1.96SE；不同shard不一定IID。
20组仍使用同一个起点与同一val集，不能消除固定val取样的共同偏差。

单独运行分支：

```bash
torchrun --standalone --nproc_per_node=2 train_fresh_repeat.py --experiment elr001 --replicate-id 0 --fresh-shard-index 23 --resume /runs/shared/step016000 --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/pulse23 --log-root outputs/logs --run-name pulse23
```

例index23需在checkpoint当前shard之后。首次分叉替换旧prefetched batch；本版中途恢复
保持replicate-id/shard-index/experiment，换新checkpoint输出目录，保留保存的游标与batch。
旧版fresh-shard续训的中途checkpoint应使用生成它的helper版本；原shared16k输入兼容不变。

开发检查是可选独立脚本：python verify.py；python verify_checkpoint.py。
验证范围：Python3.10/3.12静态检查，固定seed的shard顺序、20组40作业、配对统计，
tinyCUDA顺序读取/rank分区及完整恢复。未读取用户真实16k checkpoint，未启动正式训练。
