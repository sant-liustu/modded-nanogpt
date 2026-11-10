# 修正版Nesterov与Polyak补充8组

| GPU | 方法 | 全局BS | beta | ELR | 实际峰值eta |
|---|---|---|---|---|---|
| 0 | nesterov | 64 | .98 | 0.96 | 0.0192 |
| 1 | nesterov | 64 | .98 | 1.92 | 0.0384 |
| 2 | nesterov | 64 | .98 | 3.84 | 0.0768 |
| 3 | nesterov | 128 | .98 | 1.92 | 0.0384 |
| 4 | nesterov | 128 | .98 | 3.84 | 0.0768 |
| 5 | nesterov | 128 | .98 | 7.68 | 0.1536 |
| 6 | polyak | 128 | .98 | 7.68 | 0.1536 |
| 7 | polyak | 32 | .98 | 3.84 | 0.0768 |

## 更新定义
全部6份Nesterov均从已验证修正后BS32模板生成。d_t=theta_t-theta_{t-1}，两个theta均为完成投影后的base；初始previous=theta0。y=theta_t+beta*d_t，前瞻不投影；在同一y完成所有backward；精确恢复base，clip50，再raw=theta_t+beta*d_t-eta*g(y)，最后投影矩阵。previous_parameter在原地更新前写入当前base，checkpoint携带该状态；不保存/递推raw velocity。gamma使用同一参数差分但不投影。Polyak在base求梯度并用同一定义的参数差。

## 共同训练设置
沿用上批：width768/depth12/heads6/context1024/vocab50304；123,590,400参数，tied embedding/head；矩阵固定初始RMS，gamma不投影；所有参数用指定优化器。20TPP=2,471,808,000有效targets；5%线性warmup后constant，无decay；clip50包含gamma。seed0与相同训练数据tape顺序，不循环数据；FP32参数，CUDA BF16 autocast，无WD/compile/DDP。head/CE以1024tokens分块并反向重算。
BS32:75434更新/3772 warmup；BS64:37717/1886；BS128:18859/943。BS128使用microbatch64、累积2次；BS64/32使用microbatch64/32、不累积。每次完整累积后仅裁剪、更新、投影一次；Nesterov两次微批始终在同一个前瞻点。验证microbatch32，每500步及初始化/warmup结束/最终，固定262144 targets。

## Linux八卡运行
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/corrected8_20261110 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/corrected8_20261110
```
默认GPU0-7，一组一卡；--gpus可选其他卡或更少卡分波。先全部preflight再启动。每份train文件独立内嵌模型/优化器，不依赖其他本地模块。必须使用新输出目录，不把旧velocity版结果或checkpoint作为本批恢复点。

## 全尺寸显存探针
```bash
CUDA_VISIBLE_DEVICES=0 python probe_memory.py
```
默认BS128 Nesterov ELR7.68，完整模型合成token两步，包含动量状态；不读取真实训练数据、不保存模型。BS128显存探针同样使用microbatch64、累积2次，完整形状尚未实测。

## 检查范围
verify_8.py检查全部8组配置/预算、Python3.10语法、除parser外模板AST一致、tiny CUDA更新、clip50、fixednorm、恢复后下一步逐位一致与launcher dry-run。verify_displacement.py对全部6个Nesterov入口，在CPU/CUDA按独立投影参数差分参考验证变化LR、同点累积、异常恢复、previous_parameter和resume，并区分raw velocity算法。
本地Windows/Python3.12.3/PyTorch2.5.1+cu124小模型检查不等于云端Linux/Python3.10.16训练或完整显存/收敛验证。高ELR可能恶化或发散；未启动正式训练。报告见verification.json和displacement_verification.json；来源见manifest，修改见checks.diff。
