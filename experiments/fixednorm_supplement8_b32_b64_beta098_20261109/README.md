# 补充8组：batch32 / batch64，全部momentum=.98

| GPU | optimizer | batch=microbatch | effective LR | actual eta |
|---|---|---|---|---|
| 0 | nesterov | 32 | 0.24 | 0.0048 |
| 1 | nesterov | 32 | 0.48 | 0.0096 |
| 2 | nesterov | 32 | 0.96 | 0.0192 |
| 3 | nesterov | 32 | 1.92 | 0.0384 |
| 4 | polyak | 64 | 0.48 | 0.0096 |
| 5 | polyak | 64 | 0.96 | 0.0192 |
| 6 | polyak | 64 | 1.92 | 0.0384 |
| 7 | nesterov | 64 | 0.48 | 0.0096 |

有效LR=eta/(1-beta)。每份train文件独立内嵌模型和优化器；不依赖其他本地模块。
沿用昨日低显存fixednorm协议：width768/depth12/heads6/context1024/vocab50304；123,590,400唯一参数；20TPP=2,471,808,000 targets。seed0、相同训练tape顺序、不重复数据。矩阵投影回初始RMS，gamma不投影，所有参数均用指定优化器。clip50包含gamma。FP32参数、BF16 autocast，无WD/compile/DDP。输出head/CE以1024tokens分块并在反向重算。
b32:75434更新、3772 warmup；b64:37717更新、1886 warmup；之后constant。验证batch固定32，与b32源模板一致，每500步及初始化/warmup结束/最终验证262144 targets。

## Nesterov版本
明确使用后创建的batch32正确参数差分模板：d_t=theta_t-theta_{t-1}，两个theta都是已完成投影的base。前瞻y=theta_t+beta*d_t，前瞻不投影；整个梯度累积在同一个y，恢复base后clip50，raw=theta_t+beta*d_t-eta*g(y)，最后投影矩阵。previous_parameter在step前保存当前base，checkpoint携带此状态。不递推raw velocity。所有5份Nesterov入口一致。

## Linux启动
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/supplement8_beta098 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/supplement8_beta098
```
默认GPU0-7，一组一卡；--gpus可选其他卡或更少卡自动分波。先对全部候选preflight，再启动。输出目录按optimizer/batch/LR/beta分开。使用新output-root；旧velocity或其他源码checkpoint不能直接恢复。

## 验证
python verify_8.py：配置/预算、Python3.10语法解析、与昨日模板除parser外函数AST一致、tiny CUDA三步、全局裁剪、固定范数、恢复下一步逐位一致、8卡launcher dry-run。
python verify_displacement.py：5个Nesterov入口CPU/CUDA独立参数差分公式对照，含变化LR、同点累积、previous base状态、异常恢复、resume及与raw velocity的区别。
本地Windows/Python3.12.3/PyTorch2.5.1+cu124检查已通过；没有启动正式训练，未验证云端Linux/Python3.10.16、全尺寸显存或收敛。报告见verification.json和displacement_verification.json。源路径见manifest，差异见checks.diff。
