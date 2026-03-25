## Pending Task

### 1. Skill: Lecture Curation

Goal:
- 针对单个 lecture（对应一个 Python 文件），从多份相关资料中提出一个讲义策划方案，决定哪些内容应当纳入讲义，以及应当如何组织 section。
- 该 skill 的目标不是为整个 deep learning 领域设计总框架，而是为当前这一个 lecture 做内容取舍与结构设计。
- 该 skill 默认不是一次性定稿，而是先给出方案，再通过与用户的交互迭代收敛。

When to use:
- 当用户为某一个 lecture 提供多份资料，希望决定讲什么、不讲什么、以及怎么分 section 时。
- 当当前 lecture 主题边界不清、资料很多、需要先做内容策划时。

Artifact flow:
- In: lecture topic / target file + related materials
- Out: lecture plan

Responsibilities:
- 读取并拆解与当前 lecture 相关的多份课程 PDF、讲义、笔记或其他参考资料。
- 比较这些资料在当前主题上的共识、差异和取舍。
- 结合项目目标与读者画像，决定哪些内容应保留，哪些内容应删除，哪些内容应降级为补充阅读。
- 为当前 lecture 设计 section 结构与内部顺序，而不是照搬任意单一来源的目录。
- 对每个 section 给出一句话目标，并说明它与前后 lecture 的衔接关系。
- 在用户没有明确提供足够资料时，可以少量补充外部资料以弥补明显缺口，但补充资料应服务于当前 lecture 的策划，而不是扩张主题范围。
- 先给出一个可讨论的 lecture plan 草案，再根据用户反馈修改，直到结构与取舍达成一致。

Non-goals:
- 不为整个 deep learning 课程设计总目录。
- 不直接写教程正文。
- 不进入具体 code block / text block 设计。
- 不做 section 内部的详细内容填充。

Inputs:
- 当前 lecture 对应的主题或 Python 文件。
- 多份与该 lecture 相关的课程 PDF、讲义、论文、笔记或其他资料。
- 项目目标、读者画像与写作约定。

Outputs:
- 一份针对当前 lecture 的内容策划文档。
- 明确的 section 划分、section 顺序、每个 section 的一句话目标。
- 保留 / 删除 / 补充阅读建议。
- 当前 lecture 与前后 lecture 的衔接说明。
- 一份可供用户确认和修改的方案，而不是默认终稿。

Suggested artifact:
- `specs/lectures/<lecture_name>_plan.md`

### 2. Skill: Skeleton Drafting

Goal:
- 基于已经确定的单个 lecture 结构计划，结合课件、模型知识和必要搜索，完成该 lecture 的 skeleton 填充。
- 输出面向 notebook 教程编写的 lecture 骨架，而不是一次性写完所有正文。

When to use:
- 当 lecture plan 已经基本确定，需要把它展开成 notebook 级别的章节骨架时。
- 当用户需要 text block / code block 层面的结构草案，但还不准备写最终成稿时。

Artifact flow:
- In: lecture plan
- Out: lecture outline / skeleton

Responsibilities:
- 以 lecture 计划文档为约束，逐 section 提炼应保留的核心概念。
- 给出该 lecture 的 text block / code block 主结构。
- 标注哪些位置需要最小可运行实验，哪些位置只需要文字推进，哪些位置可以留给补充阅读。
- 必要时补充课件之外但对项目目标重要的知识点。
- 让该 lecture 服务于“科研上手”的目标，而不是停留在课程复述。

Non-goals:
- 不重新做 lecture 层面的内容取舍，除非发现 lecture plan 存在明显缺口并单独汇报。
- 不默认生成最终成稿，优先产出可继续写作的 skeleton。
- 不最终决定具体代码块写什么；如果需要具体演示代码，由 Skill 4 负责。

Inputs:
- 单个 lecture 的计划文档。
- 多份与该 lecture 相关的课程 PDF。
- 项目目标与写作约定。
- 必要时的补充搜索结果。

Outputs:
- 当前 lecture 的 skeleton。
- 面向 notebook 的 text block / code block 结构建议。
- 最小可运行实验位置建议。
- section 级别的教学推进顺序。

Suggested artifacts:
- `specs/lectures/<lecture_name>_outline.md`

### 3. Skill: Writing Revision

Goal:
- 对已有讲义草稿进行 revise，使其在整体写作气质上达到“深入浅出、由具体到抽象、由问题到机制”的要求。
- 该 skill 关注写作逻辑与表达质量，而不是机械套用固定模板。

When to use:
- 当 lecture 已经有 skeleton 或初稿，但讲解不够顺、太像资料堆砌、或不够“深入浅出”时。
- 当需要统一多个 section 的教学节奏、叙述逻辑和读者体验时。

Artifact flow:
- In: lecture outline / draft
- Out: revised draft

Responsibilities:
- 检查一节内容是否沿着读者的认知路径推进，而不是沿着作者的知识结构堆叠。
- 调整讲解顺序，使内容尽量遵循“问题或现象 -> 机制直觉 -> 必要抽象 -> 验证或应用”的默认推进方向。
- 删除或压缩只回答“是什么”、却没有回答“为什么需要”或“解决了什么问题”的段落。
- 将抽象内容尽量绑定到最小例子、可运行代码、图示或反例中的至少一种。
- 在不牺牲严谨性的前提下，提升文字的可读性、推进感和教学性。
- 保证不同章节可以采用不同展开方式，但整体上保持一致的写作逻辑与教学风格。
- 可以指出某处“需要一个例子或演示”，但不负责最终决定是否插入 code block，也不负责生成最终代码块。

Non-goals:
- 不机械模仿某一教材的句式、段落模板或表面文风。
- 不强制每个小节使用完全一致的结构。
- 不以“覆盖更多信息”为目标；优先保证理解路径清晰。
- 不作为最后一步去插入或删除具体代码块；代码演示的最终决策由 Skill 4 负责。

Inputs:
- 已有讲义草稿。
- 项目目标、读者画像与写作约定。
- 必要时的原始课件、补充材料或相关代码。

Outputs:
- 修订后的章节或小节草稿。
- 对主要修改点的简要说明。
- 对仍然存在的理解断层、术语跳跃或示例不足处的提示。

Suggested artifacts:
- `specs/writing_logic.md`
- `specs/revision_checklist.md`

Instruction template:
- 写作应以认知推进为核心，而不是以知识罗列为核心。
- 优先回答“为什么会有这个问题”与“这个机制解决了什么”，再回答“它的正式定义是什么”。
- 讲解顺序应尽量遵循：问题或现象 -> 机制直觉 -> 必要抽象 -> 验证或应用。
- 这不是固定模板；当内容不适合完整走这条路径时，也应至少保证读者知道：当前在解决什么问题、为什么需要这个概念、以及这个概念如何落到代码、实验或系统现象上。
- 抽象概念的讲解，必须尽量绑定到最小例子、可运行代码、图示或反例中的至少一种。
- 正式定义、公式和术语应在读者已经形成初步直觉后引入。
- 一次只引入当前问题所需的最小新概念，避免在单个小段中叠加过多新抽象。
- 代码块必须服务于理解，不只是展示实现。
- 每一段文字都应当有推进性，让读者比上一段多看清一层机制。
- 允许不同章节采用不同展开方式，但整体上应保持“由具体到抽象、由现象到机制、由局部到系统”的叙述方向。

Negative constraints:
- 不要从定义列表开始。
- 不要为了完整而过早展开旁支内容。
- 不要把名词解释写成百科词条。
- 不要把 code block 当作展示实现能力的堆砌。
- 不要默认读者会自动把数学、代码和系统现象联系起来；需要显式搭桥。
- 不要让一节内容只回答“是什么”，却不回答“为什么重要”。

### 4. Skill: Demo Insertion

Goal:
- 在单个 lecture 中判断哪些位置应插入演示代码块，以及哪些位置不应插入。
- 插入的代码块必须服务于理解，风格极简，接近“可运行的伪代码”，而不是工程化实现。
- 该 skill 是 lecture 编写流程的最后一步之一，用来决定最终是否需要 code block，以及 code block 具体如何出现。

When to use:
- 当 lecture 的文字结构和写作逻辑已经基本稳定，需要决定最终哪些地方插入代码演示时。
- 当某些 section 需要通过极简代码来建立直觉、验证机制或展示现象时。

Artifact flow:
- In: revised lecture draft
- Out: demo plan / minimal code blocks

Responsibilities:
- 判断某一段概念是否值得通过代码演示来建立直觉、展示现象或验证机制。
- 只在代码能明显提升理解时插入 code block；如果文字、图示或公式已经足够清楚，则不插入。
- 为适合演示的概念设计最小可运行代码块，优先展示机制而不是完整系统。
- 让代码块与前后的 text block 紧密配合，使读者知道代码在观察什么、为什么值得运行。
- 尽量把每个代码块组织成若干个短小函数，避免把逻辑散落在长脚本中。
- 默认少写代码块；只有在代码能显著增加理解时才写。

Non-goals:
- 不为了“看起来像 notebook”而机械增加代码块数量。
- 不写工程化、生产化或可复用性优先的实现。
- 不引入防御性编程、配置系统、日志系统、命令行接口、类层次或多余抽象。
- 不把 tutorial code block 写成完整库函数或项目模块。
- 不扩展 lecture 的理论覆盖范围。

Decision rules:
- 当代码能够帮助读者观察一个现象、理解一个机制、比较两种行为、或验证一个推导时，优先插入代码块。
- 当内容主要是定义、历史背景、术语整理、经验总结或高层框架时，通常不插入代码块。
- 当某段内容若没有代码就会显得空泛，或者读者运行一个极小例子后能立刻获得直觉时，应考虑插入代码块。
- 如果代码块不能明显带来新的观察，而只是把文字重复翻译成实现，则不要插入。

Code style constraints:
- 代码必须极简，像能运行的伪代码。
- 优先使用若干个短函数，而不是一个大而全的脚本。
- 只保留理解当前机制所需的最小参数、最小数据和最小控制流。
- 不写防御性分支、边界检查、异常处理、日志、配置对象、命令行入口、测试脚手架或性能优化样板。
- 不为了“规范”而增加类封装、抽象基类、工厂模式或其他冗余设计。
- 变量名、函数名和输入都应服务于教学清晰度，而不是工程通用性。

Inputs:
- 单个 lecture 的 skeleton 或草稿。
- 当前 lecture 的 section 目标与上下文。
- 必要时的相关资料与参考实现。

Outputs:
- 建议插入代码块的位置清单。
- 每个代码块的教学目的说明。
- 极简代码块草稿。
- 不建议插入代码块的位置与原因。

Suggested artifacts:
- `specs/lectures/<lecture_name>_demos.md`
- `specs/lectures/<lecture_name>_code_blocks.md`

### Workflow Boundary

- Skill 1 负责单个 lecture 的资料筛选、内容取舍与 section 结构设计。
- Skill 2 负责在既定 lecture plan 下进行 skeleton 填充。
- Skill 3 负责对已有草稿进行写作逻辑与教学风格上的 revise。
- Skill 4 负责在最后阶段判断何时插入演示代码块，并生成极简的 tutorial code blocks。
- Skill 1 先出方案，再通过用户交互收敛，不默认一次性定稿。
- Skill 1 不写教程正文。
- Skill 2 不重做 lecture 层面的内容取舍，除非发现明显缺口。
- Skill 3 可以建议需要例子或演示，但不最终决定 code block 的插入与实现。
- Skill 4 不负责扩展 lecture 的理论覆盖范围；它只决定是否需要代码演示，以及代码应如何以极简形式出现。
