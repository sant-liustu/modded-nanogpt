# stable16000：20个全新shard上的配对重复（不打乱token）

每组从同一个w768/d12 shared-stable16000完整checkpoint恢复，选择一个此前未读过
的独立训练shard，从该shard开头按原token顺序续训。同组两条分支共享同一shard；
20组使用20个不同shard。默认2卡、每卡B64、全局B128；不重置模型、Adam状态、
scheduler、初始矩阵RMS目标或模型RNG。保留norm control及gamma LR0.0018。

| 分支 | 调度 | 新增更新数 | intrinsic time |
|---|---|---:|---:|
| baseline | 始终0.06 | 520 | 31.2 |
| elr001 | 0.01跑120步，再0.06跑500步 | 620 | 31.2 |

## 怎样保证没有使用过

只接受之前已核实的shared-stable16000版本source/schedule SHA256。
读取checkpoint保存的原排序shard列表、文件大小/mtime、rank/microbatch、global batch
和游标，检查所有文件仍一致。重放原loader从第0步到16000步的全部microbatch取数，
共16000×gradient accumulation+1次（包括保存的prefetched next batch）。
重放结果必须与checkpoint保存的current_shard/current_position完全一致。

凡已训练、部分训练、预取或当前预加载的shard都整份排除；最后一项略保守。
如果历史曾循环遍历全部shard，就没有可证明未用过的原列表shard，直接报错。
自动从剩余shard中按原排序选20个，每份必须能独立容纳620步及预取batch。
默认B128/context1024要求至少81,395,713 tokens/份；不拼接、不溢出、不回卷到旧shard。
不足20份或历史游标无法核实会拒绝运行，不静默复用数据。

注意“没用过”的证明仅针对这份checkpoint所对应的、已固定源码的训练链，
不能证明它没有在其他外部预训练任务里见过同一文本，或不同shard内容不重复。

## 运行

保持原checkpoint的数据glob、路径/mtime、rank、microbatch、PyTorch/CUDA与compile设置。
不要把input-bin改成新shard单文件；通过fresh-shard-index在原列表内选择。
先在训练机器检查历史排除列表、选出的20份shard和40条命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_repeats.py --resume /runs/shared/step016000 --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --output outputs/fresh_shard20 --dry-run
```

dry-run会在CPU以mmap读取rank0 checkpoint元数据和各shard头，不训练、不输出模型数据。
去掉--dry-run开始40次顺序续训，复用两张GPU；默认--repeats20。
可用--fresh-shard-indices显式指定20个原列表下标，但同样必须通过未使用/长度检查。
全部结束后自动计算配对均值；任一作业失败即停并保留已有输出。

单独一组（例index23必须是审计确认未用过且足够长的shard）：

```bash
torchrun --standalone --nproc_per_node=2 train_fresh_repeat.py --experiment elr001 --replicate-id 0 --fresh-shard-index 23 --resume /runs/shared/step016000 --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/pulse_shard23 --log-root outputs/logs --run-name rep0_pulse
```

只在历史16k首次分叉时换shard，并丢弃旧prefetched batch。中途恢复本版checkpoint时
保持replicate-id/fresh-shard-index/experiment，使用新checkpoint输出目录；恢复已保存
的当前位置和prefetched batch，不回到shard开头。错误配对ID、shard、helper版本或调度拒绝。
最终保留baseline终点、pulse120步窗口末尾和pulse终点，共60份完整多rank checkpoint。

## Val与平均

起点及每次更新后都测同一固定val前缀，默认1,048,576 tokens/次。
实际验证由探针每步评估分支控制；为恢复旧checkpoint保留的config.val_loss_every=0不关闭这里的验证。
--probe-val-tokens可统一加大以检查验证集取样敏感性，不能在同一配对内混用。

先逐组计算delta=L_pulse-L_baseline，再对20个delta取平均。
共同tau网格为0,.06,...,31.2，共521点。baseline第i步对应pulse第6i步(i<=20)，
之后对应pulse第i+100步，完全精确对齐，不用插值。

```bash
python aggregate_pairs.py --run-root outputs/fresh_shard20
```

输出：aggregate/per_pair_delta.csv、mean_delta.csv和summary.json。
包括均值、跨shard的sample SD、SE和描述性mean±1.96SE逐点区间。
检查缺失点/非有限loss、同组不同shard、不同组复用shard、shard已在历史排除名单中、
起点val不一致等错误。下载后保留manifest.json和logs目录的相对结构即可重新汇总。

新shard可降低对单条未来训练轨迹的依赖，但不同shard不一定IID，误差区间仅作描述。
20组仍条件于同一个16k模型，并且使用同一验证集；不会消除固定验证集的共同取样偏差。
同物理步两组batch相同，相同tau时两组消耗的batch/Adam/gamma更新次数仍不同。
此前那两次运行不自动混入这20组均值。

## 验证

python verify.py
python verify_checkpoint.py

前者仅标准库，检查继承代码、游标重放、旧/部分/预加载/长度不足shard拒绝、
20个不同新shard/40作业、实际调度停止和已知答案的配对统计。
后者用真实合成bin和tinyCUDA模型，重放实际旧loader16001次取数，验证新shard连续读取、
rank分区、拒绝回卷、初始val/RNG一致以及40次完整状态迁移和中途恢复。
这些是本地合成检查；未读取用户真实16k checkpoint，未启动正式数据训练。
