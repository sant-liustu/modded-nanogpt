# Batch32：SGD四组 + Polyak四组

| 默认GPU | 优化器 | effective LR | momentum | 实际峰值eta |
|---|---|---|---|---|
| 0 | SGD | .06 | 0 | .06 |
| 1 | SGD | .12 | 0 | .12 |
| 2 | SGD | .24 | 0 | .24 |
| 3 | SGD | .48 | 0 | .48 |
| 4 | Polyak | .24 | .9 | .024 |
| 5 | Polyak | .48 | .9 | .048 |
| 6 | Polyak | .96 | .9 | .096 |
| 7 | Polyak | 1.92 | .9 | .192 |

每份train_*.py为独立单文件，内嵌模型和优化器，无其他本地模块依赖。
LR与beta在对应文件中固定；effective LR=eta/(1-beta)，不是可学习RMSNorm gamma。
Polyak继续显式使用d_t=theta_t-theta_{t-1}，参数是完成投影后的base；gamma不投影。

统一设置：global batch32=microbatch32，不做梯度累积；width768/depth12/heads6/
context1024/vocab50304，fanin_fixed初始化。矩阵固定初始RMS，矩阵和gamma都用
指定优化器，无辅助Adam。全局L2梯度裁剪50，包含gamma，日志保留裁剪前范数。
FP32参数、CUDA BF16 autocast，无WD/compile/DDP；seed0，同一训练数据顺序。
低显存head/CE默认每1024tokens分块，反向checkpoint重算；Transformer整批32。

123,590,400唯一参数，20TPP=2,471,808,000有效targets；75,434更新，前3,772步
线性warmup（5%向上取整），之后恒定LR，无尾部decay。最后一步19,456targets。
每500步及初始化/warmup结束/最终测验证，共153次；每次固定262144targets。
本次相对batch32模板只改候选LR/beta约束及默认独立输出路径，原有实验目录不变。

## 八卡启动

```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/sgd_polyak8_b32 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/sgd_polyak8_b32
```

默认GPU0–7，一组一卡；可通过--gpus指定更少GPU并分波运行。
启动前检查全部数据/config；各组console、checkpoint和metrics目录独立。
同版本相同命令恢复latest.pt；其他协议或旧版本checkpoint请勿直接复用。

## 单独运行

```bash
CUDA_VISIBLE_DEVICES=0 python train_sgd_b32_elr0p06_beta0p0.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output runs/sgd_b32_006
CUDA_VISIBLE_DEVICES=4 python train_polyak_b32_elr1p92_beta0p9.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output runs/polyak_b32_192
```

可先运行`CUDA_VISIBLE_DEVICES=0 python probe_memory.py`，以完整模型形状和合成
token执行两步显存探针；不读取真实训练数据、不保存模型。尚未执行全尺寸测试。

verify_8.py已覆盖八组参数/预算、函数AST与模板一致、候选实际LR下tiny CUDA更新、
全局clip50及完整累积后裁剪、optimizer恢复后下一步逐位一致和启动dry-run。
报告见verification.json，全部候选差异见checks.diff。测试模板在tests/，正式训练
不依赖它。Windows/Python3.12小模型CUDA检查不代表云端Linux/Python3.10训练、
完整768显存或收敛验证。未启动正式训练。
