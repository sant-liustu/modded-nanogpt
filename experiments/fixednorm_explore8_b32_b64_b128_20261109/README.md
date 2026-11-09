# 8组补点与BS128探索

| GPU | 方法 | BS=microbatch | beta | ELR | 实际峰值eta |
|---|---|---|---|---|---|
| 0 | polyak | 64 | 0.98 | 3.84 | 0.0768 |
| 1 | nesterov | 32 | 0.98 | 3.84 | 0.0768 |
| 2 | sgd | 128 | 0.0 | 0.12 | 0.1200 |
| 3 | sgd | 128 | 0.0 | 0.24 | 0.2400 |
| 4 | sgd | 128 | 0.0 | 0.48 | 0.4800 |
| 5 | polyak | 128 | 0.98 | 1.92 | 0.0384 |
| 6 | polyak | 128 | 0.98 | 3.84 | 0.0768 |
| 7 | nesterov | 128 | 0.98 | 0.96 | 0.0192 |

模型width768/depth12/heads6/context1024/vocab50304；123,590,400参数，共2,471,808,000训练targets=20TPP。矩阵固定初始RMS，gamma不投影，全部参数使用所选优化器，无辅助Adam。clip50包含gamma；seed0、固定训练tape顺序；无WD/compile/DDP；FP32参数、CUDA BF16 autocast。head/CE以1024tokens分块并反向重算。验证microbatch固定32，与上一批相同。
BS32:75434更新/3772 warmup；BS64:37717/1886；BS128:18859/943。warmup后constant。各脚本默认microbatch等于全局batch，无梯度累积。最后一步精确mask预算尾部。
Nesterov沿用后创建的BS32参数差分版：d=theta_t-theta_prev，两者均为投影后base；y=theta_t+beta*d，所有backward在同一y；恢复base、clip、raw=theta_t+beta*d-eta*g(y)、投影矩阵。前瞻点不投影，不递推raw velocity。

## Linux八卡运行
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/explore8 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/explore8
```
默认GPU0-7一组一卡；--gpus可改映射或使用更少卡分波。先检查全部数据和配置。新实验使用新的output-root，不兼容旧源码checkpoint。每份train文件可独立运行，不依赖本地其他模块。

## BS128显存探针
```bash
CUDA_VISIBLE_DEVICES=0 python probe_memory.py
```
默认用完整BS128 Polyak模型和合成token跑两步，含动量状态，不启动真实训练。可用--script选择其他候选。BS128整批激活显存比BS64高，完整形状尚未实测。

## 验证
python verify_8.py及python verify_displacement.py已通过。包含8组预算/参数、模板函数AST一致、Python3.10语法解析、tiny CUDA三步/clip/fixednorm/恢复下一步逐位一致，2个Nesterov入口CPU/CUDA独立参数差分公式与异常恢复，以及launcher dry-run。
这是Windows/Python3.12.3/PyTorch2.5.1+cu124的小模型验证，不等于云端Linux/Python3.10.16或全尺寸显存/收敛验证；未启动正式训练。BS128动量候选搜索不完整，不能据此比较各方法最优值。来源见manifest，差异见checks.diff。
