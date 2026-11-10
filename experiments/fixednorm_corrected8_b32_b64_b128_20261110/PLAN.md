用户指定8组，全部beta=.98。修正版Nesterov显式使用连续投影后base参数之差，不递推raw velocity。控制模型/数据/seed0/20TPP/clip50/低显存head与上批相同；5%warmup后constant。仅改batch、候选ELR、输出路径。检查预算、AST一致、tiny更新、范数、恢复和独立参数差分参考。高ELR可能恶化或发散，不预设排序；无全尺寸云端验证，不启动正式训练。

用户补充：所有全局BS128入口使用microbatch64累积2次，预算和warmup以全局更新计数不变。
