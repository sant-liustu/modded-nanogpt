# BS256/512 八组实验

| 默认GPU | 方法 | 全局BS | microbatch | 累积次数 | beta | ELR | 实际峰值eta |
|---|---|---|---|---|---|---|---|
| 0 | sgd | 256 | 64 | 4 | 0.0 | 0.24 | 0.24 |
| 1 | sgd | 256 | 64 | 4 | 0.0 | 0.48 | 0.48 |
| 2 | polyak | 256 | 64 | 4 | 0.98 | 3.84 | 0.0768 |
| 3 | polyak | 256 | 64 | 4 | 0.98 | 7.68 | 0.1536 |
| 4 | nesterov | 256 | 64 | 4 | 0.98 | 1.92 | 0.0384 |
| 5 | nesterov | 256 | 64 | 4 | 0.98 | 3.84 | 0.0768 |
| 6 | sgd | 512 | 64 | 8 | 0.0 | 0.48 | 0.48 |
| 7 | sgd | 512 | 64 | 8 | 0.0 | 0.96 | 0.96 |

共同协议：width768/depth12/heads6/context1024/vocab50304，123,590,400唯一参数；20TPP=2,471,808,000有效targets。BS256共9430更新、472步warmup；BS512共4715更新、236步warmup；5%线性warmup后constant，无decay，最后更新按有效token精确截断。seed0，同一训练数据tape顺序、不循环数据。矩阵固定初始RMS，gamma不投影，所有参数用指定优化器；全局clip50包含gamma，在完整累积后只裁剪一次、step一次、投影一次。
Nesterov使用修正版投影后base参数差分d=theta_t-theta_prev；前瞻y=theta_t+beta*d不投影；四个microbatch始终在同一y完整backward，之后精确恢复base、clip、raw=theta_t+beta*d-eta*g(y)、投影；previous_parameter在原地更新前保存当前base，不递推raw velocity。
FP32参数、CUDA BF16 autocast，无WD/compile/DDP；低显存head/CE分块1024tokens并反向重算。验证microbatch32、固定262144targets，每500步及初始化/warmup结束/最终验证，与前批保持同一更新间隔。

## 八卡运行
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/eight_b256_b512_20261113 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/eight_b256_b512_20261113
```
默认GPU0-7，一组一卡；--gpus可指定实际卡号或更少卡分波。先全部preflight再训练；每份train独立内嵌模型/优化器。必须使用新输出目录，不恢复不同batch或源码checkpoint。

## 检查与限制
verify_8.py验证参数/预算/Python3.10语法/模板函数AST一致、tiny CUDA对应4次或8次累积、完整累积后clip、固定范数、恢复后下一步逐位一致及八卡dry-run。verify_displacement.py对两个Nesterov做CPU/CUDA独立参数差分参考，含变化LR、多microbatch同一前瞻、previous base、异常恢复和resume。
CUDA_VISIBLE_DEVICES=0 python probe_memory.py：默认完整BS256 Nesterov/microbatch64四次累积，合成tokens两步显存探针，不读真实数据或保存模型。
本地Windows/Python3.12.3/PyTorch2.5.1+cu124小模型检查不等于云端Linux/Python3.10.16或全尺寸显存/收敛验证。未启动正式训练；本次仅生成，不推送。来源见manifest，修改见checks.diff，报告见verification.json/displacement_verification.json。
