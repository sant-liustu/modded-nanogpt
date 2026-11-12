用户确认linear rule两批各8组：BS8/4各方法两点、BS2各方法一个点，另BS128 Nesterov2.88。参考BS16 SGD .12/.24、Polyak .96/1.92、Nesterov .48/.96，ELR随BS线性缩放；BS2选各参考下点。参考不是确认最优；BS1282.88为指定补点。沿用20TPP/fixednorm/clip50/seed0/warmup_constant/低显存；Nesterov投影后base参数差分。BS128 micro64累积2次。检查预算/AST/tiny CUDA/恢复/独立差分；推送但不启动真实训练。BS8 SGD两点已有，不重复启动。

用户修正候选（覆盖此前linear参考选点）：BS2 SGD .03、Polyak .24、Nesterov .24；BS8 Polyak/Nesterov各.48/.96；BS4 Polyak/Nesterov各.24/.48。其他候选不变。
