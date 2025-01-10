# RL Based on Dynamic Programing

## 策略迭代

对于一个动作空间和状态空间有限的 MDP，策略迭代过程以下面的顺序进行：

- 随机初始化策略 $\pi$
- 重复以下过程直到收敛
  - 计算 $V \coloneqq V^\pi$
  - 对于每个状态，更新
    $$\pi(s)=\arg\max_{a\in A}r(s,a)+\gamma\sum_{s'\in S}P_{sa}(s')V(s')$$
