# 默认执行顺序

本仓库的默认处理顺序固定为：

`lecture-curation -> lecture-drafting -> lecture-revision -> 验证 -> 更新 progress.md`

## 第 1 步

`lecture-curation`

先定讲义结构、章节边界和内容安排。

## 第 2 步

`lecture-drafting -> spawn rough-drafter subagents`

按章节拆分，使用 `rough-drafter` 并行产出粗稿，然后由主 agent 合并。
图片首次插入也放在这一步，按章节一起分配给对应的 `rough-drafter`。

## 第 3 步

`lecture-revision -> spawn writing-reviser subagents`

按章节拆分，使用 `writing-reviser` 并行修订，然后由主 agent 做整讲整合。
这一阶段只整理、移动或清理已经存在的图片，不负责首次找图。

## 收口

`主 agent 合并 -> 验证 -> 更新 progress.md`

如果用户明确要求提交，再进入 commit 流程。
图片路径默认写成相对目标文档的路径；只有目标文档位置变化，或者图片资源被换到别的目录时，才需要同步改路径。
