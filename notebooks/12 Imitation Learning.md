# Imitation Learning

## 1 Introduction to Imitation Learning

传统强化学习任务十分依赖奖励函数的设置，但是在很多现实场景中，可能会出现难以明确即时奖励的情况，例如自动驾驶、计算机游戏等，此时随机设计的奖励函数将无法保证强化学习训练出的策略满足实际需要。

*模仿学习*（imitation learning）目的是模仿专家智能体的决策行为，借此绕开传统强化学习对奖励信号的要求。

目前模范学习方法大致可以分为三类：

- 行为克隆（behavior cloning）
- 逆强化学习（inverse RL）
- 生成对抗式强化学习（generative adversarial imitation learning，GAIL）

![imitation learning](../image/12.1.png)

一般来说，模仿学习的智能体有以下性质：

- 可以取得（非交互的）专家智能体的轨迹数据
- 可以与环境或模拟器交互
- 无法取得奖励信号

![imitation learning](../image/12.2.png)
