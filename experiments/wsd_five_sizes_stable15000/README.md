# 五尺寸 WSD：先跑 stable 到15000步

配置：矩阵ELR在1–924步线性warmup到0.06，925–15000步保持0.06。
默认完成15000次更新，保存checkpoint后退出，不进入decay。
内部完整时间轴仍为18858步；名义后续linear decay为3858步。
Gamma绝对LR0.0018，924步warmup后恒定；不随矩阵decay。
全局B128、context1024、seed0、AdamW、WD0、每步固定初始RMS。
五模型分别为w384/d12、w768/d6、w768/d12、w768/d24、w1536/d12。
实际执行训练tokens为1,966,080,000；五模型完全一致。

## 八卡运行

从本目录在Linux/CUDA训练机执行（替换数据路径，输出必须是新目录）：

```bash
python launch.py --train-glob '/data/fineweb_train_*.bin' --val-glob '/data/fineweb_val_*.bin' --output outputs/wsd_stable01 --dry-run
python launch.py --train-glob '/data/fineweb_train_*.bin' --val-glob '/data/fineweb_val_*.bin' --output outputs/wsd_stable01
```

GPU0–3运行w1536/d12；GPU4–7先运行w768/d24，完成后GPU4–5运行
w768/d12，GPU6运行w768/d6，GPU7运行w384/d12。显存未做正式尺寸实测。

单独运行w768/d12：

```bash
torchrun --standalone --nproc_per_node=2 train_fixednorm_w768.py --input-bin '/data/fineweb_train_*.bin' --input-val-bin '/data/fineweb_val_*.bin' --checkpoint-dir outputs/w768_ckpt --log-root outputs/logs --run-name w768_wsd
```

查看预算：`python train_fixednorm_w768.py --describe-only`。

## Checkpoint与恢复

自动保存step005000、step010000、step015000。每个目录的complete.json
在所有rank写入完成后生成。保留模型、两个optimizer、scheduler、初始RMS目标、
Python/NumPy/CPU/CUDA RNG、train/val loader位置和下一batch。
若某个作业失败，可单独恢复到新的checkpoint输出目录（卡数与配置须相同）：

```bash
torchrun --standalone --nproc_per_node=2 train_fixednorm_w768.py --resume outputs/w768_ckpt/step010000 --checkpoint-dir outputs/w768_resume --log-root outputs/resume_logs --run-name w768_resume
```

默认仍在15000停止。`--stop-step`可设置更早保存停止点；后续可用相同源文件
恢复。显式`--stop-step 18858`会执行预置的3858步linear decay，这不是默认行为。
任意新decay曲线尚未实现；恢复仍严格校验源码、配置、调度、数据文件和rank数。
不要用旧linear脚本的checkpoint恢复本实验。

验证默认关闭，保持旧尺寸实验协议；训练loss及原有每1000步诊断保留。
本次只准备stable checkpoint；后续探针的固定验证集评估应在分叉脚本中加入。
相对旧linear组，矩阵warmup从1886改为924；gamma warmup仍为924。

## 验证

`python verify.py`：五尺寸meta参数量、完整18858步调度、配置保护、
非调度函数AST与旧脚本一致、tiny CPU三步AdamW/投影逐位回归、八卡launcher dry-run。
`python verify_checkpoint.py`：tiny CUDA模型完整状态捕获/恢复和下一步逐位比较。
这些检查不代表正式尺寸DDP/编译训练已经完成。未启动正式训练。
来源与SHA256见provenance.json；设计与审查要点见PLAN.md。

发布说明：verify_checkpoint.py可在新机器独立执行；verify.py的旧源码回归检查需要provenance.json记录的历史本地源文件路径，其结果已保存在verification.json。训练脚本、launch.py及checkpoint验证不依赖这些历史路径。
