# w768/d12：intrinsic-time等宽脉冲与训练阶段检验

从stable checkpoint5000、10000、15000、16000各自独立分叉六组。
每个分叉直接恢复指定checkpoint，不额外预训练；不会让一个脉冲分支成为另一个的起点。
默认2卡、每卡B64、全局B128、context1024。保留AdamW全部状态、固定初始RMS
投影、gamma绝对LR0.0018、训练数据位置、预取batch及全部RNG。

| experiment | 脉冲ELR | 脉冲更新数 | 恢复0.06更新数 | 总新增更新数 | 总intrinsic time |
|---|---:|---:|---:|---:|---:|
| baseline | 始终0.06 | — | — | 520 | 31.2 |
| elr005 | 0.05 | 24 | 500 | 524 | 31.2 |
| elr004 | 0.04 | 30 | 500 | 530 | 31.2 |
| elr003 | 0.03 | 40 | 500 | 540 | 31.2 |
| elr002 | 0.02 | 60 | 500 | 560 | 31.2 |
| elr001 | 0.01 | 120 | 500 | 620 | 31.2 |

所有脉冲的intrinsic width均为1.2；恢复段为30。
基线前20步对应intrinsic time1.2，之后500步覆盖同一恢复时间范围。
例如fork5000、elr005：5001–5024使用0.05；5025–5524使用0.06。
脉冲结束和最终步保存完整checkpoint；baseline在最终步保存。

## 验证与日志

默认从分叉起点、每次更新后均测val loss（固定每步，不能调稀），包括切换点和终点。
每次使用同一验证文件前缀1048576 tokens；可用--probe-val-tokens统一修改。
这个数是全体GPU的总验证token数，不是每卡预算。须能整除验证global microbatch token数。
新评估协议存在cli字段中，保持历史训练config不变以进行受控checkpoint迁移；
原config中val_loss_every=0不控制本探针的每步评估。

loss.txt保留train/val loss；probe_metrics.jsonl包含step、fork_step、branch_step、
intrinsic_time（自分叉点起）、recovery_intrinsic_time（脉冲组tau-1.2，窗口内为负）、
experiment、pulse_elr、pulse_updates、val_tokens、val_loss、next_elr。
起点先测val再执行第一次训练更新，可补齐原来未测的baseline验证轨迹。
520步baseline含521个验证点；24个分支合计13176次训练更新、13200次验证。

## 一条命令运行四个checkpoint × 六组

在Linux/CUDA机器上，从本目录执行，替换四个真实checkpoint路径；先dry-run核对：

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_probes.py --checkpoints /runs/w768/step005000 /runs/w768/step010000 /runs/w768/step015000 /runs/shared/step016000 --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --output outputs/intrinsic_pulses01 --dry-run
```

删掉--dry-run开始执行。24个作业顺序使用相同两卡；任何作业失败即停止。
允许只给任意子集的checkpoint。输出根目录必须是新目录。
manifest.json保存每条命令，console目录保存每个作业控制台日志。

## 单独分支

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_elr005.sh --fork-step 5000 --resume /runs/w768/step005000 --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/f5000_elr005 --log-root outputs/logs --run-name f5000_elr005
CUDA_VISIBLE_DEVICES=0,1 bash run_baseline.sh --fork-step 5000 --resume /runs/w768/step005000 --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/f5000_baseline --log-root outputs/logs --run-name f5000_baseline
```

其余run_elr004.sh、run_elr003.sh、run_elr002.sh、run_elr001.sh用法一致。
也可直接torchrun train_intrinsic_pulse.py并指定--experiment。
不读数据/GPU查看预算：

```bash
python train_intrinsic_pulse.py --fork-step 16000 --experiment elr001 --describe-only
```

## Checkpoint来源和同分支恢复

5000/10000/15000接受088c926的w768/d12完整checkpoint；16000接受上一套
共享stable1000步后生成的checkpoint（source及stable schedule SHA256均固定）。
全部仍严格核对模型/训练config、optimizer顺序、norm targets、PyTorch/CUDA、
rank数、microbatch、compile设置、数据文件路径/大小/mtime及loader状态。
不接受不同模型、旧linear、任意改源码生成或错误阶段的checkpoint。
本次未检查实际用户checkpoint文件；如果来源不符会明确拒绝，不静默放松校验。

同分支恢复：保持原始--fork-step和--experiment，--resume指向本分支中途checkpoint，
--checkpoint-dir换新目录。保存的probe_fork_step和验证token数也会核对。
--save-at与--stop-after-save可提前保存停止，save-at不能超过本分支终点。
不要修改源码后直接恢复同分支checkpoint。

## 比较方式与限制

在相同fork点用L_branch(tau)-L_baseline(tau)构造响应，按eta-0.06归一化。
不同分支的窗口内采样tau网格不同，需插值或只比较共同网格；不可按global step直接相减。
恢复段各组tau=1.2+0.06k与baseline第20+k步恰好对齐，不需要插值。
跨checkpoint比较时各自先扣除本checkpoint基线，再比较相同局部tau。
固定验证前缀消除样本选择变化，但不是独立重复实验；幅度缩放不等于叠加性已经验证。
相同tau下训练tokens、Adam/gamma更新次数不同，是解释线性/时不变结果时的混杂因素。

## 本地验证

python verify.py
python verify_checkpoint.py

verify.py检查24套实际CLI配置、全部调度及停止块、继承训练定义token SHA256、
launcher同起点依赖、固定前缀每步验证和预算；不依赖历史源文件绝对路径。
verify_checkpoint.py在tiny CUDA eager模型上测试24套受控迁移、切换实际矩阵LR、
gamma LR、完整磁盘保存恢复后下一步loss/模型/optimizer/scheduler/RNG逐位一致，
以及错误源码/调度/分叉点/分支/验证预算拒绝。loader使用合成测试对象。
报告见verification.json及checkpoint_verification.json。
未启动真实数据训练；正式尺寸DDP/compile及真实checkpoint兼容性尚未实测。

验证器兼容Python3.10与3.12：继承代码使用包含装饰器和缩进边界的token哈希，避免ast.dump新增type_params字段造成误报。保留旧AST哈希供溯源；训练源码、checkpoint迁移及恢复校验均未改动。
