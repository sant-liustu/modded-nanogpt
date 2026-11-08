# SGD五组 + Polyak三组：每组单GPU

八个训练文件均为独立单文件，无本地模型/优化器模块依赖。
模型、fixednorm、数据读取、更新公式和20TPP预算实现沿用上一版。
每份脚本固定effective LR、momentum和默认独立输出路径；现已统一加全局L2梯度裁剪50。
可用`--output`指定新路径；LR/beta不能被命令行改成另一候选。

| GPU（默认启动映射） | 文件 | effective LR | beta | 实际峰值eta |
|---|---|---|---|---|
| 0 | train_sgd_elr0p03.py | .03 | 0 | .03 |
| 1 | train_sgd_elr0p06.py | .06 | 0 | .06 |
| 2 | train_sgd_elr0p12.py | .12 | 0 | .12 |
| 3 | train_sgd_elr0p24.py | .24 | 0 | .24 |
| 4 | train_sgd_elr0p48.py | .48 | 0 | .48 |
| 5 | train_polyak_elr0p06.py | .06 | .9 | .006 |
| 6 | train_polyak_elr0p12.py | .12 | .9 | .012 |
| 7 | train_polyak_elr0p24.py | .24 | .9 | .024 |

effective LR=eta/(1-beta)，与RMSNorm可学习gamma区分。
矩阵和gamma均使用同一所选优化器。全局batch64=microbatch64，无梯度累积。
width768/depth12/heads6/context1024/vocab50304；共享embedding/head只计一次，
123,590,400参数，训练2,471,808,000有效tokens=20TPP。
37,717次更新，前1,886次warmup（5%向上取整），之后恒定直到结束。
最后一次52,224有效tokens。每500步及warmup/结束验证固定262144 targets。
无WD/compile/DDP；FP32参数、CUDA BF16 autocast。
梯度累积完成后、optimizer.step之前，对所有参数（包括一维gamma）的全局L2梯度
执行clip_grad_norm_(max_norm=50,error_if_nonfinite=True)。日志gradient_norm保留裁剪前值。
裁剪的是梯度，不是momentum速度或最终参数位移。源码/contract已改变，使用新输出目录。
不下载或循环训练数据。

## Linux八卡并行

```bash
python launch_8.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/search8 --dry-run
python launch_8.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/search8
```

可用`--gpus 0,1,2,3,4,5,6,7`更改八张卡的顺序/ID。每个子进程只暴露一张卡，
不使用torchrun；先检查八组数据量和配置，然后并行训练。
各组console日志在output-root/console，checkpoint和验证日志在各自run子目录。
相同命令可恢复同版本checkpoint；新LR/模型/协议使用新目录。

## 单独运行任一组

```bash
CUDA_VISIBLE_DEVICES=0 python train_sgd_elr0p03.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output runs/sgd03
CUDA_VISIBLE_DEVICES=5 python train_polyak_elr0p06.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output runs/polyak06
```

单文件本身不硬编码物理GPU编号，通过CUDA_VISIBLE_DEVICES选择；默认用可见的
cuda:0，仅占一张卡。不传--output则写入runs/对应候选名称。

## 本地验证与审查

verify_8.py：八份文件预算/固定参数、模型/优化器AST与上一版一致、
各候选实际LR下tiny模型三次CUDA更新、warmup/constant、fixednorm、
Polyak前一参数state恢复，以及八卡launcher dry-run；不用真实训练数据。
checks.diff列出每个文件相对于原模板的完整修改。
结果见verification.json。tiny模型使用小词表、宽度及单层，不能代表完整
768模型microbatch64的显存/稳定性或Linux/Python3.10训练验证。
未启动正式训练，未自动选择或增加其他候选。
