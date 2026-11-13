# BS512 Polyak与Nesterov：四组两卡DDP

| 默认GPU | 方法 | beta | ELR | 实际峰值eta |
|---|---|---|---|---|
| 0,1 | Polyak | .98 | 3.84 | .0768 |
| 2,3 | Polyak | .98 | 7.68 | .1536 |
| 4,5 | Nesterov | .98 | 3.84 | .0768 |
| 6,7 | Nesterov | .98 | 7.68 | .1536 |

每组两卡DDP，每卡microbatch64累积4次，全局BS512。4715次更新，前236次线性warmup后constant；20TPP=2,471,808,000有效targets，最后一步精确mask。width768/depth12/heads6/context1024/vocab50304，123,590,400唯一参数；seed0，相同数据tape不重复。矩阵固定初始RMS，gamma不投影；全局梯度clip50包含gamma，完整DDP累积后裁剪、更新、投影各一次。FP32参数、BF16 autocast，无WD/compile。head/CE分块1024tokens反向重算，验证rank0固定262144targets然后广播。

Nesterov使用修正后的投影base参数差分theta_t-theta_prev，整个累积在同一个未投影前瞻点，之后恢复base、clip、更新和投影。不递推raw velocity。模型与优化器及DDP函数相对上一批不变，仅改候选、global batch和独立输出。

## 四组八卡启动
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/four_b512_ddp2 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/four_b512_ddp2
```
默认GPU0-7按两卡一组，同时运行四组。--gpus可更改卡号或指定更少的偶数张卡分波。先全部双进程preflight再训练，必须用新输出目录，不兼容旧单卡checkpoint。

## 单独运行
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_nesterov_b512_elr3p84_beta0p98.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output runs/nesterov_b512_384_ddp2
```
每个训练文件独立内嵌模型/优化器，不依赖本地其他模块。

## 显存探针
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 probe_memory.py
```
默认BS512 Nesterov7.68，完整形状合成tokens两步，不读取真实数据或保存checkpoint。

## 已验证范围
verify_ddp.py：4组真实两进程CPU/Gloo与原单卡实现比较三步loss/梯度/参数；每卡四次累积、尾步5个有效tokens且一个rank为零、跨rank参数逐位一致、保存恢复后继续更新。全部模型/优化器/DDP/验证/保存函数与上一批已测实现AST一致，仅parser候选不同；语法和八卡launcher dry-run通过。
本地Windows/Python3.12.3/PyTorch2.5.1+cu124 CPU/Gloo检查，不是云端Linux/Python3.10.16双GPU NCCL、BF16完整模型的显存/速度/收敛验证。正式训练未启动。报告见ddp_verification.json与launcher_verification.json，修改见checks.diff。
