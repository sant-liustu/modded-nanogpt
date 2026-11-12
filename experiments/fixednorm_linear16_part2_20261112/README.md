# 最终分组 第2批：8组

| GPU | 方法 | BS | 更新次数 | warmup |
|---|---|---|
| 4 | 603469 | 30174 |
| 8 | 301735 | 15087 |
| 128 | 18859 | 943 |

第一批为小batch（BS2五组+BS4三组）；第二批为大batch（BS4三组+BS8四组+BS128一组）。BS8 SGD不再列入本轮，旧实验继续沿用原脚本。候选按用户最终指定，非最优保证。
共同协议：width768/depth12/heads6/context1024/vocab50304；123,590,400唯一参数，20TPP=2,471,808,000有效targets；seed0、相同数据tape顺序、不循环数据；5%线性warmup后constant，无decay。矩阵固定初始RMS，gamma不投影；所有参数用所选优化器；clip50包含gamma。FP32参数、BF16 autocast，无WD/compile/DDP；head/CE分块1024tokens并反向重算。验证microbatch32，固定262144targets，每500步及初始化/warmup结束/最终验证。
Nesterov均用修正版d_t=theta_t-theta_prev（投影后base），y=theta_t+beta*d_t不投影；完整累积在同一个y，恢复base、clip、raw=theta_t+beta*d_t-eta*g(y)、矩阵投影；previous_parameter更新前保存当前base，不递推raw velocity。BS128使用microbatch64累积2次，其余batch=microbatch。

| BS | 更新次数 | warmup |
|---|---|---|
| 4 | 603469 | 30174 |
| 8 | 301735 | 15087 |

## 八卡启动
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/linear16_part2_20261112 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/linear16_part2_20261112
```
默认GPU0-7；--gpus可指定实际空闲卡号，先全部preflight再训练。各脚本独立内嵌模型/优化器，无本地模块依赖。分组移动导致source/output变化，已运行的候选继续从原脚本和原目录恢复，不要用这次重排后的脚本恢复旧checkpoint。未启动候选使用新输出目录。
检查：verify_8.py验证预算/参数/Python3.10语法/模板AST、tiny CUDA/裁剪/投影/恢复及八卡dry-run；verify_displacement.py对全部Nesterov在CPU/CUDA独立差分公式核对，含变化LR、多microbatch同点前瞻、previous base、异常恢复和resume。probe_memory.py完整形状合成tokens显存探针。
本地Windows/Python3.12.3/PyTorch2.5.1+cu124小模型验证，不等于云端Linux/Python3.10.16、完整显存或收敛验证；未启动正式训练。报告见verification.json/displacement_verification.json，来源见manifest，差异见checks.diff。
