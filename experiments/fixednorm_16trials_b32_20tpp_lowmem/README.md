# Batch32：第一批8组 + 第二批8组，共16个独立单文件

| 优化器 | momentum | 有效LR候选 | 数量 |
|---|---|---|---|
| SGD | 0 | .03 / .06 / .12 / .24 / .48 | 5 |
| Polyak | .9 | .06 / .12 / .24 / .48 | 4 |
| Nesterov | .9 | .12 / .24 / .48 / .96 / 1.92 | 5 |
| Nesterov | .98 | .96 / 1.92 | 2 |

effective LR=eta/(1-beta)；实际峰值eta在manifest.json逐组列出。候选没有随batch
自动缩放，仍是用户指定的16组。每份脚本固定自己的LR/beta及默认独立输出路径。

## 统一设置

- global batch32、microbatch32，不做梯度累积，一组仅用一张可见GPU。
- width768/depth12/heads6/context1024/vocab50304；最近fanin_fixed初始化和模型内嵌。
- 唯一参数123,590,400，tied embedding/head只计一次；20TPP=2,471,808,000 targets。
- 总75,434次更新，前3,772次warmup（5%向上取整），之后constant，无末尾decay。
- 最后一次19,456有效tokens；尾部mask和梯度有效token加权保留。
- 每500步及初始化/warmup结束/训练结束验证，共153次，每次固定262144 targets。
- clip50：全局L2梯度范数，包含gamma；日志gradient_norm为裁剪前值。
- 矩阵更新后投影回初始RMS，gamma不投影；所有参数均使用指定优化器，无辅助Adam。
- seed0、固定训练tape顺序；无WD、compile或DDP，FP32参数、CUDA BF16 autocast。
- 所有16组统一使用低显存输出loss：1024tokens一块，head/CE在反向时checkpoint重算。
  Transformer本身一次处理整批32条序列，不是降低microbatch。

Nesterov已修正为显式d_t=theta_t-theta_{t-1}（投影后的base参数），前瞻与更新
使用同一参数差；不保存raw velocity。Polyak同样使用参数差分。前瞻点不额外投影。
Nesterov训练loss在lookahead处，验证在base处。低显存版BF16浮点结果不保证与
原全logits实现逐位一致。旧batch64/velocity checkpoint不能直接恢复。

## 按可用GPU数量分波运行

八卡分两波，每卡同时只运行一组；第一波完成后启动第二波：

```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/search16_b32 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/search16_b32
```

有16张卡可一波跑完：

```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/search16_b32 --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

也可指定更少GPU，脚本自动分波；一张卡同时仅一个进程。启动前检查全部配置和
数据量；输出、console、checkpoint按候选名称分开。某波有失败则不启动下一波，
检查对应console日志后用相同命令恢复。未启动正式训练。

## 单文件独立训练与显存探针

```bash
CUDA_VISIBLE_DEVICES=0 python train_nesterov_b32_elr0p96_beta0p98.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output runs/nesterov_b32_096_098
CUDA_VISIBLE_DEVICES=0 python probe_memory.py
```

正式训练只需对应train_*.py，不需要其他本地模块；另需numpy、PyTorch和数据。
probe_memory.py使用全width768/depth12/batch32/context1024和合成token跑两步，
不保存模型，报告显存峰值；可用--script选候选。--loss-chunk-tokens可改输出块大小。

## 本地检查

verify_16.py：16组配置/预算、模型和更新函数AST一致性、实际候选LR下tiny CUDA
训练、clip50和完整累积后裁剪、optimizer恢复后下一步逐位一致，以及8卡两波dry-run。
verify_displacement.py：7个Nesterov入口在CPU/CUDA对照独立投影参数差分公式，
含变化LR、多microbatch同点前瞻、previous_parameter保存、异常恢复与断点恢复。
见verification.json和displacement_verification.json；变更见batch32.diff。
小模型检查不是完整768/batch32显存或收敛验证，也不是云端Linux/Python3.10验证。
