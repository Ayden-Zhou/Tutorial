# 函数逼近与泛化

## 1 逼近问题的提出

本节目标：
- 把第二讲的主问题从“会不会训练”收紧为“模型族能表示什么、学到的解为何可能在新样本上成立”。
- 明确这一讲是在第一讲“三个基本问题”之上，分别深入逼近与泛化，而不是重复总览。

主要参考：
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/3_approximation_2025.pdf`
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/7_generalization_2025.pdf`

### 1.1 监督学习的函数视角

### 1.2 以标量回归作为分析入口

### 1.3 逼近、优化与泛化的关系

## 2 分箱逼近的基本思路

本节目标：
- 用“分箱查表”建立对函数逼近的最小直觉。
- 让读者先看到存在性论证的力量，也看到它很快会遇到维度灾难。

主要参考：
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/3_approximation_2025.pdf`

### 2.1 Lipschitz 条件与分箱近似

### 2.2 一维分箱的误差控制

### 2.3 高维分箱与维度灾难

### 2.4 分箱构造的启发与局限

## 3 ReLU 网络的构造性逼近

本节目标：
- 把近似论证从离散分箱过渡到 ReLU 组合。
- 让读者理解“网络能模拟局部块状函数”这件事，而不是死记通用逼近结论。

主要参考：
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/3_approximation_2025.pdf`

### 3.1 由 ReLU 差分构造局部 bump

### 3.2 由一维局部块到多维局部区域

### 3.3 局部 bump 的叠加与整体近似

### 3.4 构造性证明的教学价值

## 4 通用逼近的含义与边界

本节目标：
- 澄清“能逼近”只是存在性结果，不等于“容易学到”或“会泛化”。
- 把第一讲里略提过的“表示能力问题”展开成更精确的判断标准。

主要参考：
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/3_approximation_2025.pdf`
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/7_generalization_2025.pdf`

### 4.1 通用逼近定理的存在性含义

### 4.2 宽度增长与大权重问题

### 4.3 查表式逼近与表征学习的区别

### 4.4 逼近能力不是学习成功的充分条件

## 5 深度与宽度的表达效率

本节目标：
- 不再停留在“加非线性就更强”，而是进一步讨论“为什么深度本身可能重要”。
- 用分段线性和 sawtooth 例子建立 depth separation 的直觉。

主要参考：
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/3_approximation_2025.pdf`

### 5.1 宽网络的优势与直觉

### 5.2 ReLU 网络与分段线性函数

### 5.3 kink 数量的增长规律

### 5.4 sawtooth 函数与深度分离现象

### 5.5 对网络设计的启示

## 6 泛化问题的基本表述

本节目标：
- 从“拟合训练集”切换到“解释新样本表现”。
- 给出一个比第一讲更具体的泛化判断框架。

主要参考：
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/7_generalization_2025.pdf`

### 6.1 经验风险与总体风险

### 6.2 记忆式解为何不是好解

### 6.3 训练拟合与真实规律

### 6.4 泛化所依赖的结构信息

## 7 经典泛化理论及其局限

本节目标：
- 给出 Occam、偏差-方差、VC 这类经典故事的最小版本。
- 重点说明这些工具为什么不足以解释现代深度网络。

主要参考：
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/7_generalization_2025.pdf`

### 7.1 Occam 偏好与模型复杂度

### 7.2 假设类大小与泛化界直觉

### 7.3 随机标签实验的挑战

### 7.4 double descent 现象的启示

## 8 深度学习中的归纳偏置

本节目标：
- 把“泛化解释”从硬性的容量控制，转向“归纳偏置从哪里来”。
- 为后续讲优化、架构与预训练埋下动机。

主要参考：
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/7_generalization_2025.pdf`

### 8.1 架构归纳偏置

### 8.2 参数到函数映射中的简单性偏置

### 8.3 优化过程带来的隐式偏置

### 8.4 从容量控制到概率偏好

### 8.5 本讲的理论边界与后续衔接
