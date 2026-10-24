# 只改 model size：固定 w768 / depth12 训练配置

基准是 同仓库 `experiments/scaling_collapse_fixednorm/train_fixednorm_w768.py`
在默认 TPP20 下解析出的配置。新目录独立保存五个完整训练脚本，原目录不变。
这里固定累计训练 tokens，不固定 TPP。

| 脚本 | 宽度 | 层数 | 参数量 | 累计训练 tokens | 实际 TPP |
|---|---:|---:|---:|---:|---:|
| train_fixednorm_w384.py | 384 | 12 | 40,563,072 | 2,471,755,776 | 60.936 |
| train_fixednorm_w768.py | 768 | 12 | 123,590,400 | 2,471,755,776 | 20.000 |
| train_fixednorm_w1536.py | 1536 | 12 | 417,047,040 | 2,471,755,776 | 5.927 |
| train_fixednorm_w768_d6.py | 768 | 6 | 81,112,320 | 2,471,755,776 | 30.473 |
| train_fixednorm_w768_d24.py | 768 | 24 | 208,546,560 | 2,471,755,776 | 11.852 |

## 所有尺寸共用的设置

- 全局 batch128、context1024、18,858 updates、seed0；相同数据路径和原有 loader / shard 循环规则。
- 矩阵 peak ELR0.06；1,886 步线性 warmup，随后 16,972 步线性衰减到零。
- Gamma 绝对 peak LR0.0018；warmup924 步后恒定。
- GPU 分配按模型大小设置，始终保持全局 batch128，见下表。
- 保留 fan-in 初始化规则、固定初始 RMS、matrix/gamma WD0、原始 AdamW 参数、每步投影、编译与每1000步诊断策略。
- 验证仍默认关闭（val_loss_every=0），val_tokens=10,485,760。
- 不按模型大小缩放 ELR 或训练预算。不接受 `--tpp`；科学配置覆盖参数若偏离基准会报错。

| 模型 | GPU数 | 每卡 batch |
|---|---:|---:|
| w384 / 12层 | 1 | 128 |
| w768 / 12层 | 2 | 64 |
| w1536 / 12层 | 4 | 32 |
| w768 / 6层 | 1 | 128 |
| w768 / 24层 | 4 | 32 |

默认八卡分两路并行：GPU0–3 跑 w1536；GPU4–7 先跑 w768/24层，完成后 GPU4–5 跑 w768/12层、GPU6 跑 w768/6层、GPU7 跑 w384。分配复用此前固定TPP实验的方案，未在正式训练机上测定显存或吞吐最优值。

“只改 model size”指保持模型构造规则：head_dim128，head 数随宽度改变；
fan-in 标准差随张量形状改变，logits 仍除以 width/768，层数变化不额外加 residual multiplier。
因此固定的是 ELR 及参数化规则，实际矩阵 LR 仍为 ELR × 各矩阵保存的目标 RMS。
这与原 w768 脚本的优化器语义一致。

数据来源、shard 顺序和预处理必须对各模型使用同一套输入。相同 seed 保留原先随机数规则，
不代表不同结构逐元素共享初始化。相同 token 预算也不代表相同 FLOPs 或训练耗时。

## 使用

查看预算（不启动训练）：

```powershell
python train_fixednorm_w384.py --describe-only
```

在 Linux/CUDA 训练机上，从本目录执行：

```bash
python launch.py --train-glob '/data/fineweb_train_*.bin' --val-glob '/data/fineweb_val_*.bin' --output outputs/model_only01 --gpu-ids 0,1,2,3,4,5,6,7 --dry-run
python launch.py --train-glob '/data/fineweb_train_*.bin' --val-glob '/data/fineweb_val_*.bin' --output outputs/model_only01 --gpu-ids 0,1,2,3,4,5,6,7
```

也可单独启动某个脚本：

```bash
torchrun --standalone --nproc_per_node=4 train_fixednorm_w1536.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --run-name w1536 --checkpoint-dir outputs/w1536
```

如大模型在默认每卡 batch32 下显存不足，可降低该模型的 `--device-batch-size`，
使用继承的梯度累积机制维持全局 B128。执行参数 `expected-world-size`、
`device-batch-size`、`compile-model` 可覆盖；默认执行配置按模型尺寸变化，训练预算和科学配置保持一致。
数据/日志路径、run name 和 checkpoint 操作也可设置；日志放在新目录。
旧脚本 checkpoint 不应直接续训，新脚本保留原来的严格源码/配置校验。

## 验证与来源

发布前已在本地检查五个模型实际参数量、预算一致性、函数及模型定义一致性、全部更新步的矩阵与 gamma 调度、tiny CPU 三步 forward/backward + AdamW + 固定 RMS 投影逐位回归，以及启动器分配、GPU 映射、阶段屏障和失败阻断。
测试脚本和测试产物不包含在本发布目录中。来源和训练脚本 SHA256 见 `publication_manifest.json`。
这些检查不代表正式尺寸 GPU/DDP/Inductor 或真实数据训练已验证；没有启动正式训练。

## 补充模型：默认单组八卡

新增 `train_fixednorm_w768_d48.py`（378,458,880 参数）和 `train_fixednorm_w1536_d24.py`（756,825,600 参数）。两者均默认8卡、每卡batch16、不累积梯度，保持全局batch128、18,858步、2,471,755,776 tokens及原linear调度/优化器配置。

在本目录执行，替换为与原五组一致的数据路径；两组顺序运行：

```bash
torchrun --standalone --nproc_per_node=8 train_fixednorm_w768_d48.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --run-name w768_d48 --checkpoint-dir outputs/w768_d48
torchrun --standalone --nproc_per_node=8 train_fixednorm_w1536_d24.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --run-name w1536_d24 --checkpoint-dir outputs/w1536_d24
```

原launch.py仍只启动原五组。新增两组已通过预算、实际参数量、函数AST一致性、逐步调度、CPU三步回归和配置保护检查；未运行正式GPU/DDP训练。

## 等参数量的宽深对照（新增）

| 新脚本 | 参数量 | 对照模型 | 默认GPU数 / 每卡batch |
|---|---:|---|---|
| train_fixednorm_w384_d36.py | 83,054,976 | w768/d6：81,112,320 | 2 / 64 |
| train_fixednorm_w384_d60.py | 125,546,880 | w768/d12：123,590,400 | 4 / 32 |

两者都保留固定训练量：18,858 updates、2,471,755,776 tokens、全局B128，
矩阵peak ELR0.06、warmup1886步后linear decay；其余科学配置与原model-only组一致。
val loss仍默认关闭。初始化仍按fan-in，logits仍除以width/768，不增加深度缩放。
它们是近似等参数量的宽深对照，参数量分别高约2.40%和1.58%。
GPU分配是起始配置，未实测正式模型显存；必要时降低device batch并保留全局B128。

两个新增脚本独立启动，不自动加入原八卡五模型启动计划；可在6张空闲GPU上同时运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_fixednorm_w384_d36.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --run-name w384_d36 --checkpoint-dir outputs/w384_d36
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --standalone --nproc_per_node=4 train_fixednorm_w384_d60.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --run-name w384_d60 --checkpoint-dir outputs/w384_d60
```

上述两条命令分别在两个终端执行；输出目录应为新路径。
