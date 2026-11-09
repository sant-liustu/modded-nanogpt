# 实验计划
补齐用户指定8组；不新增搜索范围。b32 Nesterov gamma=.24/.48/.96/1.92；b64 Polyak gamma=.48/.96/1.92；b64 Nesterov gamma=.48；全部beta=.98。
沿用昨日fixednorm低显存模板，模型、数据顺序、seed0、20TPP、clip50、5%warmup+constant不变。指标为完整预算最终验证loss；没有预设优化器排序。检查候选与预算、模板函数一致性、tiny更新、投影、恢复和launcher；不在本机启动真实训练。审查重点：batch默认值、eta=gamma*(1-beta)、投影后base参数差分和独立输出目录。不同batch更新次数和warmup步数不同；本地检查不验证云端显存与收敛。
