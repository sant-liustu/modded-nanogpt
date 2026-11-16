# Nesterov BS8/16/64：8组独立单卡脚本

| 默认GPU | 全局BS=microbatch | beta | ELR | 实际峰值eta |
|---|---|---|---|---|
| 0 | 16 | 0.95 | 1.92 | 0.096 |
| 1 | 16 | 0.98 | 0.6 | 0.012 |
| 2 | 16 | 0.98 | 1.2 | 0.024 |
| 3 | 64 | 0.98 | 1.6 | 0.032 |
| 4 | 64 | 0.98 | 2.4 | 0.048 |
| 5 | 64 | 0.95 | 1.92 | 0.096 |
| 6 | 8 | 0.98 | 0.72 | 0.0144 |
| 7 | 8 | 0.98 | 1.2 | 0.024 |

ELR=eta/(1-beta)，每份候选的ELR/beta由parser固定。注意beta=.95、ELR1.92的实际峰值eta=.096；beta=.98时eta=.02*ELR。

共同训练协议：width768/depth12/heads6/context1024/vocab50304；123,590,400唯一参数，tied embedding/head。20TPP=2,471,808,000有效targets；seed0、相同训练tape顺序，不循环数据。5%更新线性warmup后constant，无末尾decay。矩阵更新后投影回初始RMS，gamma不投影；所有参数都由Nesterov更新。全局L2梯度裁剪50包含gamma。FP32参数、CUDA BF16 autocast，无WD/compile/DDP。head/CE分块1024tokens并在反向重算。验证microbatch32、固定262144targets，每500步及初始化/warmup结束/最终验证。
Nesterov严格使用修正版参数差分：d_t=theta_t-theta_prev，两个theta均为完成投影后的base，初始previous=theta0。y=theta_t+beta*d_t不额外投影；整个backward在同一前瞻点，精确恢复base后clip50，再raw=theta_t+beta*d_t-eta*g(y)，最后投影矩阵。previous_parameter在更新前保存当前base，不递推raw velocity。gamma采用相同差分但不投影。

| BS | 总更新 | warmup |
|---|---|---|
| 8 | 301735 | 15087 |
| 16 | 150868 | 7544 |
| 64 | 37717 | 1886 |

## 八卡启动
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/nesterov_eight_20261116 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/nesterov_eight_20261116
```
默认GPU0-7，一组一卡，每卡microbatch=该组全局batch，不做梯度累积。--gpus可指定其他卡号或更少卡分波。先全部preflight再训练。每份train文件都独立内嵌模型/优化器，不需要本地其他模块；依赖PyTorch、NumPy和原预分词bin数据。使用新的输出目录，不恢复不同ELR/beta/源码checkpoint。

## 单组例子
```bash
CUDA_VISIBLE_DEVICES=0 python train_nesterov_b16_elr1p92_beta0p95.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output runs/nesterov_b16_192_beta095
```

## 验证与显存探针
python verify_8.py：配置/预算/Python3.10语法/模板函数AST一致、实际峰值LR下tiny CUDA三步、clip50/固定范数/恢复后下一步逐位一致及八卡dry-run。python verify_displacement.py：8个入口CPU/CUDA独立参数差分参考，覆盖两种beta、变化LR、同点累积、previous base、异常恢复、checkpoint恢复并区分raw velocity。
CUDA_VISIBLE_DEVICES=0 python probe_memory.py：默认BS64/ELR2.4完整模型合成tokens两步显存探针，不读取数据或保存checkpoint。
本地Windows/Python3.12.3/PyTorch2.5.1+cu124小模型验证不等于云端Linux/Python3.10.16、完整模型显存或收敛验证。没有启动正式训练；本次只准备本地脚本，未推送。来源见manifest，修改见checks.diff，报告见verification.json/displacement_verification.json。
