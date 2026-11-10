# BS16六组独立单文件

| GPU | 方法 | beta | ELR | 实际峰值eta |
|---|---|---|---|---|
| 0 | sgd | 0.0 | 0.06 | 0.0600 |
| 1 | sgd | 0.0 | 0.12 | 0.1200 |
| 2 | polyak | 0.98 | 0.96 | 0.0192 |
| 3 | polyak | 0.98 | 1.92 | 0.0384 |
| 4 | nesterov | 0.98 | 0.96 | 0.0192 |
| 5 | nesterov | 0.98 | 1.92 | 0.0384 |

BS16=microbatch16，无梯度累积，一组一卡。width768/depth12/heads6/context1024/vocab50304；123,590,400参数；20TPP=2,471,808,000有效targets。150,868更新、7,544步warmup后constant；最后一步3,072 targets。seed0、相同训练tape顺序，不循环数据。
矩阵投影回初始RMS，gamma不投影；所有参数用所选优化器。clip50包含gamma；FP32参数、BF16 autocast，无WD/compile/DDP。head/CE分块1024tokens并反向重算。验证microbatch32，固定262144 targets，每500步及初始化/warmup结束/最终验证。
Nesterov使用修正版：d_t=theta_t-theta_prev（两者为投影后base），前瞻y=theta_t+beta*d_t不投影；完整backward后恢复base、clip、raw=theta_t+beta*d_t-eta*g(y)、投影矩阵。previous_parameter保存更新前当前base，不递推raw velocity。gamma也采用参数差分但不投影。Polyak在base求梯度。

## 六卡启动
```bash
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/six_b16_20261111 --dry-run
python launch.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/six_b16_20261111
```
默认GPU0-5，一组一卡；用--gpus 2,3,4,5,6,7指定实际空闲卡号，也支持更少卡分波。先全部preflight再训练。新实验使用新输出目录，不兼容旧velocity/不同batch的checkpoint。每份train文件可独立运行，无其他本地模块依赖。

## 验证
python verify_6.py及python verify_displacement.py已通过：预算/参数/Python3.10语法/模板AST、tiny CUDA更新/clip/fixednorm/恢复下一步逐位一致、六卡dry-run；2个Nesterov入口CPU/CUDA独立差分公式，含变化LR、同点累积、previous base、异常恢复和区分raw velocity。
CUDA_VISIBLE_DEVICES=0 python probe_memory.py：完整BS16 Nesterov合成token两步探针，不读取数据或保存模型。
本地Windows/Python3.12.3/PyTorch2.5.1+cu124小模型验证不等于云端Linux/Python3.10.16训练、完整显存或收敛验证。未启动正式训练。候选不保证最优。报告见verification.json和displacement_verification.json；来源见manifest，修改见checks.diff。
