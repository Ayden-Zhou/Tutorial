# Demo Format

Use this reference when inserting final teaching demos into this project.

## Output Modes

Choose one mode before editing.

### Full-Lecture Mode

Use when the user wants demo insertion across a whole lecture.

Recommended pattern:

````md
## 4 某个 section 标题

前面我们已经从公式上看到宽模型可以表示这类函数，但直觉上还是有点飘。下面插一个极小实验，只看一维输入下 ReLU 分段线性的效果。

```python
import numpy as np
import matplotlib.pyplot as plt


def relu_features(*, x_grid: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.maximum(x_grid[:, None] - centers[None, :], 0.0)


x_grid = np.linspace(-1.0, 1.0, 400)
centers = np.array([-0.6, -0.2, 0.2, 0.6])
features = relu_features(x_grid=x_grid, centers=centers)
weights = np.array([1.0, -1.0, 1.0, -1.0])
y_values = features @ weights

plt.plot(x_grid, y_values)
plt.title("Piecewise linear function from ReLU features")
plt.show()
```

运行后注意看两个点：第一，函数确实是分段线性的；第二，每增加一个折点，本质上就是多引入一个激活位置。
````

### Section-Scoped Mode

Use when the user wants only one named `##` or `###` block updated.

Recommended pattern:

````md
### 5.2 Sawtooth 结构为什么适合讲 depth separation

如果只看结论，depth separation 很容易变成一句抽象口号。这里插一个极小代码块，专门看“重复组合之后折点数量如何增长”。

```python
import numpy as np
import matplotlib.pyplot as plt


def sawtooth(*, x_values: np.ndarray) -> np.ndarray:
    return 2.0 * np.abs(x_values - np.floor(x_values + 0.5))


x_values = np.linspace(-2.0, 2.0, 800)
y_once = sawtooth(x_values=x_values)
y_twice = sawtooth(x_values=y_once)

plt.plot(x_values, y_once, label="one composition")
plt.plot(x_values, y_twice, label="two compositions")
plt.legend()
plt.show()
```

这里不用追求严密证明，只需要先看见：组合一次，波形就复杂一层；组合很多次，复杂度会以一种很不“宽层友好”的方式增长。
````

## Lead-In and Follow-Up

Before a demo, include a short lead-in.

Good lead-ins usually do two things:
- state the question,
- tell the reader what to observe.

Examples:
- `下面插一个极小实验，只看决策边界怎么随特征映射变化。`
- `先别急着记结论，我们先用一段最小代码看训练误差和测试误差为什么会分叉。`
- `这里真正需要观察的不是数值大小，而是曲线形状怎么变。`

After a demo, add a short follow-up when needed.

Examples:
- `运行后最重要的观察不是峰值位置，而是整体趋势已经和公式里的预测一致。`
- `如果这里两条曲线几乎重合，说明我们刚才讨论的偏置在这个设置下还没有出现。`

## Code Shape

Follow these rules:
- Prefer one small code block over multiple fragmented blocks.
- Keep imports minimal and local.
- Use helper functions only when they make the mechanism easier to read.
- Keep parameter lists tiny.
- If a figure is the natural output, generate it inside the demo block.
- Avoid notebook noise such as seed utilities, argument parsing, timers, logging, or file I/O unless the section truly depends on them.

## Project Code Conventions

When relevant, follow the project-wide conventions from `.harness/implement.md`:
- Code should look like runnable pseudocode.
- Avoid defensive programming.
- Add type hints to nontrivial helper functions.
- Use keyword arguments on the main data flow.
- For tensor shape changes, add trailing shape comments.
- Prefer vectorized tensor ops over loops across tensor dimensions.

## Placeholder Cleanup

When converting a placeholder into a real demo:
- remove the old `code block placeholder`,
- keep or rewrite the surrounding prose so it reads naturally,
- do not leave editor-facing notes in the lecture body.

## Non-Insertion Cases

If you decide not to insert a demo in a place that previously looked tempting:
- remove weak placeholder text, or
- leave a short collaborator-facing note outside the lecture body if later coordination still matters.

Do not leave a reader-facing sentence that says a demo might be added later.
