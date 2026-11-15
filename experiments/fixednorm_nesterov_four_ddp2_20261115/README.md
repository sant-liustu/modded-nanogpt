# Nesterov 四组：两卡DDP

| 默认GPU | 全局BS | ELR | beta | 梯度裁剪 | 每卡microbatch | 每卡累积 |
|---|---|---|---|---|---|---|
| 0,1 | 128 | 2.88 | .98 | 关闭 | 64 | 1 |
| 2,3 | 256 | .96 | .98 | 50 | 64 | 2 |
| 4,5 | 512 | .96 | .98 | 50 | 64 | 4 |
| 6,7 | 512 | 1.92 | .98 | 50 | 64 | 4 |

只有BS128/ELR2.88关闭梯度裁剪；仍计算并记录全局梯度范数，max_norm=inf不缩放有限梯度，非有限梯度仍报错。其他三组完整DDP累积后全局L2裁剪50。所有组保留矩阵投影回初始RMS；不clip不代表取消fixednorm。gamma不投影。

模型width768/depth12/heads6/context1024/vocab50304；123,590,400唯一参数，20TPP=2,471,808,000有效targets。seed0、同一训练tape顺序，不重复数据。BS128 18859更新/943 warmup；BS256 9430/472；BS512 4715/236。5%线性warmup后constant，无decay，尾步精确mask。FP32参数、CUDA BF16 autocast，无WD/compile；head/CE分块1024tokens反向重算。验证rank0固定262144targets，之后广播。

Nesterov均为修正版：d_t=theta_t-theta_prev（投影后的base参数）；前瞻y=theta_t+beta*d_t不投影；同一y覆盖全部microbatch累积，恢复base后按配置裁剪/不裁剪，再raw=theta_t+beta*d_t-eta*g(y)、矩阵投影。previous_parameter在更新前保存当前base，不递推raw velocity。

## 四组八卡启动
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/nesterov_four_ddp2_20261115 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/nesterov_four_ddp2_20261115
```
默认GPU0-7，两卡一组四组同时跑；--gpus可指定其他偶数张卡并分波。每份train可独立torchrun --standalone --nproc_per_node=2运行，不依赖其他本地模块。必须使用新输出目录；不复用clip50或旧源码checkpoint。

## 显存探针
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 probe_memory.py
```
默认BS512/ELR1.92完整模型合成token两步，不读取数据或保存模型。

## 验证
verify_ddp.py全部4候选两进程CPU/Gloo与对应单卡裁剪/不裁剪参考比较三步loss/梯度/参数，覆盖每卡1/2/4次累积、部分尾步和零token rank、跨rank参数一致、保存恢复后继续更新。verify_clipping.py使用范数约583的合成梯度，明确核对no-clip组保留梯度，另三组裁到50；初次gamma更新也核对。八卡launcher dry-run、Python3.10语法和哈希检查通过。
本地Windows/Python3.12.3/PyTorch2.5.1 CPU/Gloo测试，不是云端Linux/Python3.10.16双GPU NCCL、完整BF16模型显存/吞吐/收敛验证。未启动正式训练。报告见ddp_verification.json/clipping_verification.json/launcher_verification.json，修改见checks.diff。
