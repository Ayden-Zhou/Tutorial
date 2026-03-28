## 代码生成任务流程

- 修改代码，删除被取代的功能/特性代码
- 执行 verification
- 将改动内容写入 `./.harness/progress.md`

### 1.`.harness/progress.md` 写入规范

- `progress.md` 是相对上一个 commit 的状态增量文件。
- `## Current Status` 写当前真相，允许覆盖和更新式修改。包括主要变动、未完成和当前blcoker
- `## Message to Human` 只写需要人类介入做决定的事项。
- `## History Logs` 记录单次更新的增量，包含改动与验证结果；append only，最新更新放最上方。
- 不要把 `Current Status` 的内容原样重复到 `History Logs`；`History Logs` 只补充本次更新留下的过程证据。
- 如果某次尝试、回退或方向调整对后续判断有价值，再追加到 `History Logs`；不要把所有细碎操作都写成流水账。

### 3. 代码规范

- 代码必须看起来像是能运行的伪代码。
- 不包含任何防御性编程。

#### 3.1 Tensors

- 改变维度的操作必须加尾注释：`# [B, Seq, H]`
- 禁止对 tensor 维度写显式循环，用 `einsum` / `vmap` / batched op
- 热循环（sampler、optimizer）中用 in-place 操作省 VRAM，除非需要 Autograd
- 禁止硬编码 `.to(“cuda”)`，用 `tensor.to(device)`；创建时直接指定 `device=device`

#### 3.2 函数与命名

- 主数据流必须用 keyword arguments
- 接口必须有 type hints（`Tensor`, `int`, `float` 等）
- 变量名描述内容而非类型。Bad: `data, res, temp` / Good: `student_logits, loss_mask`

#### 3.3 错误处理

- 不用 try-except 吞错误，不写防御性 assert
- 让 PyTorch RuntimeError 直接抛出，保持代码的”伪代码”可读性

#### 3.3 注释

- Google Style Docstrings，公共类和主要函数必须写
- `Args` / `Returns` 中必须标注张量 Shape

示例：

```python
“””
计算缩放点积注意力。

Args:
    q: 查询张量。Shape: [Batch, Heads, Seq_Q, Dim]
    k: 键张量。Shape: [Batch, Heads, Seq_K, Dim]

Returns:
    注意力输出。Shape: [Batch, Heads, Seq_Q, Dim]
“””
```
