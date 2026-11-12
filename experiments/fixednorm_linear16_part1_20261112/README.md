# Linear rule 第1批：8组

| GPU | 方法 | BS | microbatch | beta | ELR | 实际峰值LR |
|---|---|---|---|---|---|---|
| 0 | sgd | 8 | 8 | 0.0 | 0.06 | 0.06 |
| 1 | sgd | 8 | 8 | 0.0 | 0.12 | 0.12 |
| 2 | polyak | 8 | 8 | 0.98 | 0.48 | 0.0096 |
| 3 | polyak | 8 | 8 | 0.98 | 0.96 | 0.0192 |
| 4 | nesterov | 8 | 8 | 0.98 | 0.48 | 0.0096 |
| 5 | nesterov | 8 | 8 | 0.98 | 0.96 | 0.0192 |
| 6 | sgd | 2 | 2 | 0.0 | 0.03 | 0.03 |
| 7 | nesterov | 128 | 64 | 0.98 | 2.88 | 0.0576 |

## 选点依据
候选按用户最新指定值，见上表；不再按之前不同方法的BS16参考点机械缩放。BS128 Nesterov2.88保持不变，microbatch64累积2次。
第一批BS8 SGD .06/.12与此前two_sgd_b8完全重复实验条件；若已启动/完成，不要另启动，沿用已有结果。它们只是为16份配置完整列出，不是两个新独立数据点。若要复用已有checkpoint，请从原目录原脚本恢复，本批源码输出路径变化会触发strict source校验。

## 训练协议
沿用width768/depth12/heads6/context1024/vocab50304；123,590,400唯一参数，20TPP=2,471,808,000有效targets。seed0、同一训练tape顺序、不循环数据。矩阵固定初始RMS，gamma不投影；全部参数使用所选优化器。clip50包含gamma，FP32参数、BF16 autocast，无WD/compile/DDP。head/CE分块1024tokens、反向重算；验证microbatch32固定262144targets，每500步及初始化/warmup结束/最终验证。5%线性warmup后constant，最后一步精确mask。
Nesterov均为修正版：d_t=theta_t-theta_prev（两个投影后base），y=theta_t+beta*d_t不投影；在同一y完成全部累积，精确恢复base、clip、raw=theta_t+beta*d_t-eta*g(y)、矩阵投影。previous_parameter在更新前保存当前base，不递推raw velocity。BS128两次微批保持同一点前瞻。

| BS | 更新次数 | warmup |
|---|---|---|
| 2 | 1206938 | 60347 |
| 8 | 301735 | 15087 |
| 128 | 18859 | 943 |

## 八卡启动
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/linear16_part1_20261112 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/linear16_part1_20261112
```
默认GPU0-7；--gpus可改映射。先全部preflight再训练。每份train文件独立，无其他本地模块依赖。使用新输出目录。第一批若两个SGD已有运行，跳过这两份，分别启动其余六个train文件，避免重复占卡。

## 检查
verify_8.py：配置/预算/Python3.10语法/模板AST一致、tiny CUDA训练/clip/fixednorm/恢复后下一步一致、八卡dry-run。verify_displacement.py：所有Nesterov CPU/CUDA独立参数差分参考，含变化LR、同点累积、previous base、异常恢复及resume。probe_memory.py为完整形状合成token两步探针。
本地Windows/Python3.12.3/PyTorch2.5.1+cu124小模型验证不等于云端Linux/Python3.10.16、全尺寸显存或收敛验证。未启动正式训练。小batch固定20TPP更新数较多。报告见verification.json/displacement_verification.json；来源见manifest，差异见checks.diff。

用户修正候选（覆盖此前linear参考选点）：BS2 SGD .03、Polyak .24、Nesterov .24；BS8 Polyak/Nesterov各.48/.96；BS4 Polyak/Nesterov各.24/.48。其他候选不变。
