# BS8 SGD两组

| 默认GPU | 方法 | global batch | microbatch | momentum | ELR=实际峰值LR |
|---|---|---|---|---|---|
| 0 | SGD | 8 | 8 | 0 | .06 |
| 1 | SGD | 8 | 8 | 0 | .12 |

沿用上一批模型：width768/depth12/heads6/context1024/vocab50304；123,590,400唯一参数。20TPP=2,471,808,000有效targets；301,735更新，前15,087次线性warmup后constant，最后一次3072 targets。seed0、相同训练tape顺序，不循环数据。矩阵固定初始RMS，gamma不投影，全部参数使用无动量SGD。全局L2梯度裁剪50包含gamma。FP32参数，CUDA BF16 autocast；无WD/compile/DDP，无梯度累积。head/CE以1024tokens分块并反向重算。验证microbatch32、固定262144targets，每500步及初始化/warmup结束/最终验证。

## 两卡启动
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/two_sgd_b8_20261111 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/two_sgd_b8_20261111
```
默认GPU0,1；用--gpus 6,7选择实际空闲卡号。先对两组preflight再训练。一组一卡，各自独立输出。使用新的output-root；不复用其他batch/source checkpoint。每份train文件独立内嵌模型与优化器，无其他本地模块依赖。

## 验证
python verify_2.py已通过配置/精确预算/Python3.10语法/模板函数AST一致、tiny CUDA更新、fixednorm、全局clip50和恢复后下一步逐位一致及两卡dry-run。
CUDA_VISIBLE_DEVICES=0 python probe_memory.py为完整BS8模型合成token两步显存探针，不读取真实数据或保存模型。
本地Windows/Python3.12.3/PyTorch2.5.1+cu124小模型检查不等于云端Linux/Python3.10.16训练或全尺寸显存/收敛验证。未启动正式训练。来源见manifest，差异见checks.diff，报告见verification.json。
