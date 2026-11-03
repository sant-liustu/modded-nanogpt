# w768/d12：共享1000步stable轨迹，再分叉三组ELR实验

输入：088c926版本w768/d12在step015000保存的完整checkpoint。
默认2卡、每卡B64、全局B128；保留模型、optimizer moments、scheduler、
初始RMS目标、RNG、loader位置和下一batch。Gamma绝对LR仍0.0018。

## 两阶段流程

1. stable：从15000续跑1000次更新，矩阵ELR恒定0.06，在16000保存共同checkpoint。
2. 三个分支全部从这个step016000 checkpoint恢复；共享stable阶段只训练一次。

| experiment | 后续矩阵ELR | 更新数 | 最终全局step | 分支新增intrinsic time |
|---|---|---:|---:|---:|
| stable | 恒定0.06 | 1000 | 16000 | 60 |
| pulse | 0.006×200，然后0.06×2000 | 2200 | 18200 | 121.2 |
| high | 恒定0.06 | 2000 | 18000 | 120 |
| low | 恒定0.006 | 2000 | 18000 | 12 |

pulse第16001–16200次更新使用0.006；第16201次恢复0.06。
自动保存：stable16000；pulse16200和18200；high/low18000。

## 每步固定验证集loss

从初始checkpoint状态开始，每完成一次更新都测val loss，包括stable的1000步
和全部分支更新，以及切换点和终点。--eval-every固定为1，不接受更稀疏频率。
默认每次验证1048576 tokens，所有阶段始终重置到相同验证文件前缀；可统一设置
--probe-val-tokens，须整除每次全局验证batch token数。
共享stable轨迹1001个评估点（15000–16000，含两端）；high/low各2001点；
pulse2201点。分支起点16000会重复测量，便于检查各分支起始状态一致。

日志中loss.txt保留train/val loss；probe_metrics.jsonl每步记录：
step、branch_step（本阶段新增更新数）、intrinsic_time（本阶段新增）、
intrinsic_time_since15000（含共享stable的60）、experiment、val_loss、next_elr。
stable与分支分别保存在独立UUID日志目录；画图用全局step拼接即可。
每步验证会增加运行时间，验证tokens不计入训练tokens。

## 一条命令：先stable，再依次high/low/pulse

从本目录在Linux/CUDA机器运行，替换输入路径，输出必须是新目录：

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_three.py --resume /runs/old/checkpoints/w768/step015000 --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --output outputs/w768_probes_v2 --dry-run
CUDA_VISIBLE_DEVICES=0,1 python run_three.py --resume /runs/old/checkpoints/w768/step015000 --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --output outputs/w768_probes_v2
```

launcher将共享checkpoint写入outputs/w768_probes_v2/checkpoints/stable/step016000，
三个分支均读取该目录。任何阶段失败即停止。dry-run只检查输入manifest/rank文件，
显示后续命令依赖，不会提前要求未来的16000 checkpoint已经存在。
已有本版shared16000 checkpoint时，可直接传给run_three.py并用新输出目录跳过stable。

## 单独执行与恢复

```bash
torchrun --standalone --nproc_per_node=2 train_w768_d12_continue.py --experiment stable --resume /runs/old/step015000 --checkpoint-dir outputs/shared --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --log-root outputs/logs --run-name shared_stable
torchrun --standalone --nproc_per_node=2 train_w768_d12_continue.py --experiment pulse --resume outputs/shared/step016000 --checkpoint-dir outputs/pulse --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --log-root outputs/logs --run-name pulse
```

high/low替换experiment并选择新输出目录。--save-at配合--stop-after-save可提前
保存；恢复同阶段checkpoint时使用同一个experiment，所有科学/执行配置保持一致。
所有数据文件、路径、mtime、卡数、microbatch、PyTorch/CUDA和compile设置须与
checkpoint匹配。不要改原数据文件。只提供model权重无法恢复本实验。

原始15000 checkpoint只能进入stable阶段；三个分支只接受本版共同16000
checkpoint或本版同分支checkpoint。不支持上一版直接15000分叉的checkpoint。
严格检查源码和完整调度哈希，避免将其它实验误当作共同历史。
查看预算无需checkpoint：python train_w768_d12_continue.py --experiment pulse --describe-only。

## 分析边界

分支新增intrinsic time的共同范围仍为0–12（high与pulse为0–120）。
共享1000步对应intrinsic time60，已单独记录。不能直接用high/low各2000步
终点相减估计累计kernel。pulse低ELR窗口h=1.2，high参考最多覆盖恢复y=118.8，
pulse最后20次更新超出参考范围，应截断比较。共享signal/kernel仍是待检验假设，
gamma/Adam记忆以及不同token消耗可能带来混杂。

## 验证与状态

python verify.py
python verify_checkpoint.py

检查原始配置与训练定义保持一致、全部阶段调度和更新数、200/201切换、
实际停止代码块、每步验证、launcher共享checkpoint依赖和describe预算。
tiny CUDA eager检查原15000迁移、shared16000进入所有分支、错误版本/调度拒绝、
实际矩阵ELR写入及同分支恢复后下一步loss/参数/optimizer/RNG逐位一致。
报告：verification.json、checkpoint_verification.json。
未读取真实用户checkpoint，未启动正式训练，尚未验证正式尺寸DDP/compile。
