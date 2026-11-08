# 第二批八组：Polyak一组 + Nesterov七组

八份独立单文件，每份仅用一张可见GPU；模型/数据/预算/clip50与第一批保持一致。

| 默认GPU | 文件 | 有效LR | momentum | 实际峰值eta |
|---|---|---|---|---|
| 0 | train_polyak_elr0p48_beta0p9.py | .48 | .9 | .048 |
| 1 | train_nesterov_elr0p12_beta0p9.py | .12 | .9 | .012 |
| 2 | train_nesterov_elr0p24_beta0p9.py | .24 | .9 | .024 |
| 3 | train_nesterov_elr0p48_beta0p9.py | .48 | .9 | .048 |
| 4 | train_nesterov_elr0p96_beta0p9.py | .96 | .9 | .096 |
| 5 | train_nesterov_elr1p92_beta0p9.py | 1.92 | .9 | .192 |
| 6 | train_nesterov_elr0p96_beta0p98.py | .96 | .98 | .0192 |
| 7 | train_nesterov_elr1p92_beta0p98.py | 1.92 | .98 | .0384 |

有效LR=eta/(1-beta)，不是RMSNorm可学习gamma。LR与momentum在每份脚本中固定，
命令行不能改成另一候选。模型宽768、深12、6heads、context1024；最近fanin_fixed
模型内嵌，tied embedding/head只计一次。矩阵更新后投影回初始RMS；gamma不投影。
矩阵与gamma都用选定优化器，无辅助Adam、WD、compile或DDP。

全局batch64、microbatch64，无梯度累积。总123,590,400参数，20TPP=
2,471,808,000 targets；37,717更新，前1,886更新warmup，之后constant，无decay。
seed0；相同训练数据顺序。每500步及warmup结束/最终验证，共78次，
每次固定262144验证targets。梯度全局L2裁剪50，包含gamma，日志记录裁剪前范数。
参数FP32，CUDA BF16 autocast；不自动下载、重复训练数据或扩大搜索范围。

Nesterov现在显式使用d_t=theta_t-theta_{t-1}，两者都是已完成更新/投影的base参数。
初始theta_{-1}=theta_0。先y=theta_t+beta*d_t，在同一y完成完整梯度累积，
精确恢复base、全局clip50，再theta_raw=theta_t+beta*d_t-eta*g(y)，最后投影矩阵。
前瞻点不额外投影。不再保存或递推raw velocity。保存的previous_parameter在正式
更新前写入theta_t，checkpoint携带该状态。一维gamma使用同样的参数差分但不投影。
Polyak已经使用相同的d_t定义，本次未修改Polyak训练脚本。
训练loss在lookahead测量；验证在投影后的base测量。
旧velocity版本checkpoint不能直接恢复；source/config变化需用新输出目录。


## 八卡运行（Linux，在本目录）

```bash
python launch_8.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/second8 --dry-run
python launch_8.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/second8
```

默认使用GPU0–7，每个进程只暴露一张卡；可用--gpus更改映射。先检查全部配置和
数据量，再并行训练。各组console日志、checkpoint和metrics目录相互独立。
同版本同命令恢复latest.pt；新协议/代码用新目录。第一批目录不变。

## 单组运行

```bash
CUDA_VISIBLE_DEVICES=0 python train_nesterov_elr0p96_beta0p98.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output runs/nesterov096_beta098
```

正式训练只需该单文件和预分词数据，无本地模块依赖；tests/仅用于验证。
--preflight仅检查预算/数据/config，不训练。未启动正式训练。

## 检查范围

python verify_8.py：八组参数和预算、模型/更新函数AST与clip50模板一致、
各实际峰值LR下tiny模型三次CUDA更新、超阈值裁剪（含gamma）与完整累积后裁剪、
optimizer恢复后下一步逐位相同、fixednorm和八卡启动dry-run。
结果见verification.json，审查差异见checks.diff，文件哈希见SHA256.json。
本地Windows/Python3.12、tiny CUDA测试不代表云端Linux/Python3.10训练、
完整768/microbatch64显存或收敛验证。
