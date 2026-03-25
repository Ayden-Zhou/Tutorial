## Plan 模式流程

制定计划前完成以下步骤。

### 1. 范围发现

- grep 搜索所有引用被修改文件/内容的地方，将结果列为候选影响范围
- 逐一判断每个引用是否需要同步改动

### 2. 计划输出要求

计划必须包含：
- **影响文件列表**：明确列出所有需要改动的文件（包括因引用关系需同步的文件）
- **每个文件的改动内容**：说明改什么，不只是"需要更新"
- **verification 步骤**：列出执行阶段需要运行的验证命令

执行 session 将依据此计划工作，计划不完整会导致遗漏。

### 3. verification要求

- 始终先运行 `just fmt`
- 函数级 test：直接运行最小受影响的 `pytest` target（例如 `.venv/bin/pytest .tests/test_base.py -v`）。
  - 如果不存在对应测试，为被修改函数创建一个最小且聚焦的 `pytest` 测试并运行。
  - 断言必须编码行为性质，而不只是”没有异常”。
- module-level verification
  - 规则写在对应模块目录的 `/<module>/notes.md`
  - 计划中新增的verification规则要写入 `/<module>/notes.md`
