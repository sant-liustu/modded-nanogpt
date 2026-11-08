# 第二批8组：保持batch64/microbatch64的输出层显存优化

与原第二批的8个LR/beta配置一致，仍为20TPP、5%warmup后constant、clip50。
本次输出层优化保留模型参数、初始化、Transformer forward和fixednorm规则。
新增loss_chunk_tokens属性不增加参数；训练循环在forward前设定此属性。

## 原始显存峰值与修改

原forward会生成64*1024*50304的完整logits，BF16约6.14GiB，转FP32后约12.28GiB；
交叉熵和反向还会保存/生成其他同规模张量。return_logits=False在原代码中并未
避免这笔计算，只是不把logits返回给调用方。

现在Transformer仍一次处理64条完整序列，最终hidden仅在输出投影/交叉熵阶段
按token分块（默认1024tokens）。每块CE按sum累计，最后除以全部有效target数，
保留-1 mask语义。训练使用PyTorch非重入checkpoint，反向时重算每块head/CE，
避免保存所有块的巨大logits/softmax。验证无梯度时直接逐块计算。
没有通过降低microbatch或增加训练梯度累积来省显存；每次仍一次optimizer.step。

checkpoint只覆盖无随机数的head/CE，不覆盖Transformer，也不改变Nesterov在
整个forward/backward期间保持前瞻位置、随后恢复base的顺序。return_logits=True
或无targets的调用保留完整参考路径。--loss-chunk-tokens 0可在单文件中选择参考
全logits路径；正式低显存默认1024。分块和重算增加计算耗时。

数学上仍计算同一个token均值交叉熵，但分块改变浮点计算顺序/矩阵乘形状，
尤其BF16梯度不保证逐位相同。更改代码/分块设置后使用新输出目录，原checkpoint
会因严格source/config校验拒绝恢复；本次未实现跨版本checkpoint转换。

## 在80G卡先测全尺寸两步

```bash
CUDA_VISIBLE_DEVICES=0 python probe_memory.py
```

这会使用实际width768/depth12/batch64/microbatch64/context1024形状和合成token，
执行2步以包含momentum state，输出allocated/reserved峰值；不读取数据、不存checkpoint。
可指定--script选择其他候选。若仍OOM，可试--loss-chunk-tokens 512，但Transformer
激活本身不随输出chunk下降。此探针必须由80G卡执行；本地没有声称已测完整配置。

## 八卡正式运行

```bash
python launch_8.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/second8_lowmem --dry-run
python launch_8.py --train '/data/fineweb_train_*.bin' --val '/data/fineweb_val_*.bin' --output-root runs/second8_lowmem
```

默认GPU0–7，一组一卡。可统一指定--loss-chunk-tokens 512。单独训练时，每个
train_*.py仍为独立单文件，另需numpy、PyTorch和数据，不依赖本目录其他文件。

## 已执行的本地验证

verify_lowmem.py使用batch64、context16的小模型对照完整loss：
FP32 loss差约4.8e-7，梯度相对L2误差约1.3e-7；CUDA BF16 loss在该次测试相同，
梯度相对误差约0.29%。含padding mask、验证和推理路径检查。
实际词表50304、2048tokens、width128的head局部测试：额外显存峰值从
1,252,819,968 bytes降到104,105,472 bytes（该测试chunk128）。
这个局部比例不能直接外推到完整模型的总显存或默认chunk1024。

verify_8.py验证所有8组的多块checkpoint训练、裁剪50、optimizer恢复逐位一致和
启动映射。详见verification.json、lowmem_verification.json与完整memory.diff。
Windows/Python3.12/PyTorch2.5.1的tiny CUDA检查，不等于Linux/Python3.10或
完整模型在80G卡上已通过；未启动正式训练。

## Nesterov参数差分修正（两种显存版本同步）

所有7个Nesterov入口现在显式计算d_t=theta_t-theta_{t-1}，使用已投影base参数。
前瞻y=theta_t+beta*d_t；完整backward后恢复theta_t、clip50，再更新
raw=theta_t+beta*d_t-eta*g(y)，最后投影矩阵。初始previous_parameter=theta_0。
previous_parameter在step里、原地更新前保存当前base；不再递推raw velocity。
一维gamma也使用同一个参数差分定义，但仍不投影。前瞻点不投影。
Polyak已使用该定义，本次没有修改Polyak训练文件。
旧velocity checkpoint不兼容，本次不做自动转换。

verify_displacement.py独立核对原版和lowmem共14个Nesterov文件：变化LR下的
投影二次目标、多microbatch同一前瞻、clip50、previous base状态、异常恢复、
恢复后下一步逐位一致，并明确检测与raw velocity规则不同。
结果见displacement_verification.json，完整本次差异见parameter_difference.diff。
该验证脚本跨两个实验目录读取文件；打包独立下载时仅在Git仓库布局下运行该跨目录检查。
