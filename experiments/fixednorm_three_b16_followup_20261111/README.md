# BS16三个补点

| 默认GPU | 方法 | beta | ELR | 实际峰值eta |
|---|---|---|---|---|
| 0 | sgd | 0.0 | 0.24 | 0.2400 |
| 1 | polyak | 0.98 | 3.84 | 0.0768 |
| 2 | nesterov | 0.98 | 0.48 | 0.0096 |

沿用上一批BS16：global batch16=microbatch16、无梯度累积、一组一卡；width768/depth12/heads6/context1024/vocab50304；123,590,400唯一参数。20TPP=2,471,808,000有效targets，150,868次更新、7,544步线性warmup后constant，最后一步3072targets。seed0，同一数据tape顺序，不循环数据。
矩阵固定初始RMS，gamma不投影，所有参数使用指定优化器。全局L2梯度裁剪50包含gamma。FP32参数、CUDA BF16 autocast，无WD/compile/DDP；输出head/CE默认1024tokens分块并反向重算。验证microbatch32、固定262144 targets，每500步及初始化/warmup结束/最终验证。
Nesterov严格使用修正版d_t=theta_t-theta_prev（两个都是投影后base），前瞻y=theta_t+beta*d_t不投影；完整backward后精确恢复base、clip、raw=theta_t+beta*d_t-eta*g(y)、投影矩阵。更新前保存当前base为previous_parameter，不递推raw velocity。gamma同样差分但不投影。

## 三卡启动
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/three_b16_followup_20261111 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/three_b16_followup_20261111
```
默认GPU0,1,2；用--gpus指定实际空闲卡，例如--gpus 5,6,7。先全部preflight再训练。使用新的output-root，旧协议checkpoint不兼容。每份train文件独立内嵌模型与优化器，无本地模块依赖。

## 检查
python verify_3.py：配置/预算/Python3.10语法/模板AST一致、tiny CUDA训练、clip/fixednorm/恢复后下一步一致及三卡dry-run。python verify_displacement.py：Nesterov CPU/CUDA独立参数差分公式，含变化LR、同点累积、previous base、异常恢复及resume，并区分raw velocity。
CUDA_VISIBLE_DEVICES=0 python probe_memory.py 为完整模型合成token两步显存探针，不训练真实数据或保存模型。
本地Windows/Python3.12.3/PyTorch2.5.1+cu124小模型检查不等于云端Linux/Python3.10.16训练、全尺寸显存或收敛验证。此处补点依据中途结果，不能假定最终排名；未启动正式训练。来源见manifest，差异见checks.diff，结果见verification.json与displacement_verification.json。
