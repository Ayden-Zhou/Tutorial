# Progress

## Current Status
### 主要变动
- `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 当前已完成整讲 `## 1-7` 的 rough draft。
- 本次额外清理了该讲里若干处可能导致 Markdown/KaTeX 渲染不稳的绝对值与范数写法，统一优先使用 `\left|...\right|` 与 `\left\|...\right\|`。

### 未完成
- 这讲尚未进入 `writing-revision` / `lecture-revision`。

### 当前 blocker
- 无。

## Message to Human
- 无

## History Logs
### 2026-03-28
- 按用户要求继续扫 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 里可能不稳的绝对值/范数公式写法，把 3 处 `|...|` / `\|...\|` 统一改成更稳的 `\left|...\right|` / `\left\|...\right\|`：
  - `|f(x)-g(x)|\le\varepsilon`
  - `|g(x)-g(x')|\le L\|x-x'\|`
  - `\int_{[0,1]^d}|f(x)-g(x)|\,dx\le 2\epsilon`
- 这次刻意没有改 `|\mathcal{H}|` 这类集合基数写法，因为它通常渲染稳定，且语义上不是范数兼容性问题。
- 验证：
  - `sed -n '68,92p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`
  - `sed -n '140,150p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`

### 2026-03-28
- 修复 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 中另一条 display math 的渲染兼容性问题：将
  `\|f_\theta(x^{(i)})-y^{(i)}\|_2^2+\lambda \|\theta\|_2^2`
  改为
  `\left\|f_\theta(x^{(i)})-y^{(i)}\right\|_2^2+\lambda \left\|\theta\right\|_2^2`。
- 这是最小改动，只改了该公式本身，没有调整周边正文或编号。
- 验证：`sed -n '463,474p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`

### 2026-03-28
- 修复 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 中一条 display math 的渲染兼容性问题：将
  `\lvert f_\theta(x^{(i)})-y^{(i)}\rvert^{0.25}` 改为更稳的
  `\left|f_\theta(x^{(i)})-y^{(i)}\right|^{0.25}`。
- 这是最小改动，只改了该公式本身，没有调整周边正文或编号。
- 验证：`sed -n '418,432p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`
