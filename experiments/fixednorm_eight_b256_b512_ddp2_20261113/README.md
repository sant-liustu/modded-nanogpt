# 8组实验：两卡DDP，每卡microbatch64

| 方法 | 全局BS | ELR | beta | 每卡microbatch | 每卡累积次数 |
|---|---|---|---|---|---|
| sgd | 256 | 0.24 | 0.0 | 64 | 2 |
| sgd | 256 | 0.48 | 0.0 | 64 | 2 |
| polyak | 256 | 3.84 | 0.98 | 64 | 2 |
| polyak | 256 | 7.68 | 0.98 | 64 | 2 |
| nesterov | 256 | 1.92 | 0.98 | 64 | 2 |
| nesterov | 256 | 3.84 | 0.98 | 64 | 2 |
| sgd | 512 | 0.48 | 0.0 | 64 | 4 |
| sgd | 512 | 0.96 | 0.0 | 64 | 4 |

每个trial使用两个DDP rank各一张GPU，梯度在完整累积后同步（前几轮no_sync），DDP平均前按world*valid_tokens/global_step_tokens加权。各rank读取同一global tape更新中的不同窗口，不重复计数。最后一步不足全局batch时精确mask并按有效targets计权；无有效tokens的rank走零权重dummy backward，保持同步顺序。完整同步后全局clip50、optimizer.step和矩阵投影各一次。
Nesterov保留修正参数差分：d=theta_t-theta_prev（投影后base），同一前瞻y覆盖整个本地累积；backward全部结束后精确恢复base，再裁剪/更新/投影，不递推raw velocity。前瞻点不投影，gamma不投影。
模型/seed0/训练tape/20TPP/验证样本与原单卡版一致。123,590,400参数，2,471,808,000有效训练targets。BS256 9430更新/472 warmup；BS512 4715更新/236 warmup；5%线性warmup后constant。FP32参数、CUDA BF16 autocast，无WD/compile。head/CE保持1024tokens分块重算。浮点归约顺序不同，不保证与单卡逐位相同或完整训练轨迹相同。
验证由rank0在原固定262144targets上计算，然后广播结果；只有rank0写config/log/checkpoint，其余rank同步。checkpoint保存未包DDP的模型与完整优化器previous_parameter。模型无dropout，训练tape不使用随机采样；同seed初始化、各rank相同优化器状态，复制rank0 RNG状态恢复。world=2与新源码写入contract，不直接兼容旧单卡checkpoint；使用新的输出目录。

## 单组运行（Linux）
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_nesterov_b256_elr1p92_beta0p98.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output runs_ddp2/nesterov_b256_192
```

## 全批启动
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs_ddp2 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs_ddp2
```
默认8张卡，按(0,1)/(2,3)/(4,5)/(6,7)四对同时跑4组，完成后跑后4组。若有16张卡，可加--gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15一次并行8组。每个trial独立torchrun --standalone，端口自动分配。先对全部候选双进程preflight，再启动。单脚本仍独立内嵌模型/优化器。

## 显存探针
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 probe_memory.py
```
完整模型合成token两步，各rank报告峰值显存；不读真实数据、不保存模型。可用--script选候选。两卡是否加速以及实际加速比需云端测量，不承诺2倍。

## 已完成验证
verify_ddp.py：全部8候选真实两进程CPU/Gloo，与原单卡模板比较三步loss/梯度/参数，测试每rank2/4次累积、末步5个有效tokens且一个rank无token、参数跨rank逐位一致、保存读取checkpoint后继续更新。模型与优化器类/数据loader/预算公式AST与原版一致。
verify_run.py：SGD/Polyak/Nesterov三个真实run入口，双进程小模型完整预算训练、尾步、单份日志/验证、最终checkpoint与完成后resume不重复写入。verify_ddp.py覆盖训练中保存后继续更新。launcher_verification.json覆盖8/16卡映射和语法、哈希。
本地Windows/Python3.12.3/PyTorch2.5.1+cu124，双进程CPU/Gloo测试通过。未运行云端Linux/Python3.10.16双GPU NCCL、BF16完整模型显存/吞吐/收敛测试。未启动正式训练、未推送。报告见ddp_verification.json/entry_verification.json。checks.diff包含相对原单卡模板差异。
