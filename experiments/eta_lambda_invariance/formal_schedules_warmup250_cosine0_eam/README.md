# Warmup 后 E/A/M cosine 完整八组实验

这组实验检验：从训练早期改变模块 ELR 调度后，各模块干预对固定 probe loss 的效果是否近似可加。复用共用JSON runner；本目录提供8份调度JSON及启动器。

## 训练设置与八组

使用同目录上一级的 `train_gpt2_w256_muonhinit_fixednorm_jsonelr_chord_lca.py`。默认模型12层/width256，MuonH初始化、AdamW、weight decay=0、矩阵从step0固定初始RMS、gamma不投影、直线Simpson3 LCA每2步。训练默认2张卡、global batch512、seed0、总5100步。所有 trainable tensor 按实际 `lr/RMS(W)` 控制。

根据用户最新要求，8组各自固定seed=0从第0步重新训练，不加载checkpoint。Python、NumPy、Torch CPU/CUDA均使用同一个seed，数据shard排序一致。前250步调度相同，warmup结束后才出现调度差异。固定种子不等于对所有CUDA环境承诺逐位确定性，比较时保持软件、GPU数和数据一致。以下只涉及矩阵，全部49个Norm gamma保持constant（包括mlp_norm）。E是tied embedding/lm_head。

| 配置 | cosine 的矩阵组 | 启动器 |
|---|---|---|
| C | 无 | run_C.sh |
| E | embedding | run_E.sh |
| A | attention Q/K/V/O | run_A.sh |
| M | MLP fc/proj | run_M.sh |
| EA | embedding + attention | run_EA.sh |
| EM | embedding + MLP | run_EM.sh |
| AM | attention + MLP | run_AM.sh |
| EAM | 三组全部 | run_EAM.sh |

每份JSON显式包含全局更新1～5100，不依赖runner隐含调度。warmup第u步为0.03*u/250。第251～5100步，未选中的tensor保持0.03，选中的tensor为：

`0.03 * (1 + cos(pi * (u - 251) / 4849)) / 2`

第251步仍为0.03，第252步开始低于0.03，第5100步精确为0。当前选择的cosine末端是0，不是0.003。零ELR仍推进AdamW状态，不能当作冻结或删除模块。每组从JSON的第1步开始执行。

## 云端启动

需将共用JSON runner与本目录的正式文件一起同步。使用Bash启动，不要求脚本有可执行权限。在仓库根目录：

```bash
DIR=experiments/eta_lambda_invariance/formal_schedules_warmup250_cosine0_eam

# 依次从头运行全部8组，各占用可见的两张GPU
bash "$DIR/run_all.sh"

# 或选择单组，不要再重复执行上面的run_all
bash "$DIR/run_EM.sh"
```

不需要任何checkpoint，启动器不接受checkpoint参数；启动前检查共享runner默认seed=0。任一组失败，run_all停止，排查后可重新运行对应单组启动器。不会自动跳过已有实验日志，重复启动会创建新运行。

## 输出与加性检验

每组由runner写入独立 `logs/*_jsonelr_<schedule SHA前12位>_*` 目录，保存 schedule.json、schedule_metadata.json、tensor_rms_elr_history.jsonl、lca_decomposition.jsonl 等。`manifest.json` 给出组名到JSON SHA256的映射，以避免混淆。

重点比较M的四条loss增量曲线：`L_M-L_C`、`L_EM-L_E`、`L_AM-L_A`、`L_EAM-L_EA`。E、A同理。三模块预测为 `L_E + L_A + L_M - 2*L_C`，与实际L_EAM比较。必须报告完整交互曲线和单模块效果量级，不能仅用最后一点接近断言全程可加。LCA负值代表loss下降，但LCA求和不是干预加性的检验；后者用相同probe下的实际loss。只分析warmup后的累计归因时需减去250步累计值。

单一种子、cosine末端0与本训练机制限定了结论范围；交互偏离也可能是真实现象。用户review前不标为科学结论已完成。

## 本地验证

本地辅助测试（不随正式文件上传）实际调用共用runner的JSON解析函数验证8*5100*122个ELR；独立临时小模型(2层、width32、vocab64、10更新、warmup2)运行8组，核对实际optimizer LR/RMS、相同数据/RNG、初始norm targets；C和EAM额外从头连续运行，与恢复运行的完整训练状态和后缀LCA逐位比较，均通过。最新从头启动入口另核对seed=0、无resume参数、前250步JSON完全相同。验证范围为单GPU eager，尚未实际执行云端双GPU NCCL/Inductor或Bash启动器。

`build_experiments.py`可重建本目录的8份JSON、8个单组启动器和manifest；修改生成公式后必须重验。生产运行只读取生成的JSON，不调用生成器。
