## Pending Task

### Pending Infrastructure: Demo Figure Pipeline

Goal:
- 为由 demo 代码直接生成的教学图补一层统一脚手架，而不是把这部分复杂度立即压进当前 `Skill 6: Demo Insertion`。
- 支持在 lecture 需要 loss 曲线、决策边界、attention heatmap 或其他代码生成图时，完成“运行程序 -> 记录数据 -> 统一画图风格 -> 保存到 `images/<part name>/` -> 再插回讲义”的闭环。

Why pending:
- 如果要把这件事做得可复用，仅靠当前 lecture 内联 code block 还不够；通常还需要统一的运行入口、数据记录格式、命名规则、图片落盘路径和绘图 style。
- 这已经更像一层实验/可视化基础设施，而不是当前 `Demo Insertion` 里顺手补几行 `matplotlib`。
- 在没有这层脚手架之前，`Skill 6` 仍以“直接插入最小可运行 demo，必要时允许最小 figure-from-code”作为主职责，而不是默认承担稳定的图表产线。

Likely requirements later:
- 一个轻量但统一的运行入口，用来执行 demo 并产出可复现结果。
- 一个极简数据记录约定，用来保存 loss、accuracy、boundary grid、attention map 等绘图输入。
- 一套统一的 plotting style，避免每讲各自画出风格漂移很大的图。
- 明确图片输出目录与命名约定，例如 `images/<part name>/`。
- 讲义中的 image block 与生成脚本、输出图片之间的可追溯关系。

Current decision:
- 暂不为这件事新建独立 skill，也暂不扩展当前 `demo-insertion` 到完整图表产线。
- 先把它记为 pending infrastructure；等这类需求足够多时，再决定是扩展 `Skill 6` 还是额外抽出一层专门脚手架。

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

### 2. Skill: Rough Draft

Goal:
- 基于已经确定的单个 lecture 结构计划，结合课件、参考资料和必要搜索，直接写出一版材料尽量完整的 lecture 粗草稿。
- 该 skill 的目标是先把值得保留的内容真正写进草稿，而不是只产出 skeleton；结构、语气和教学节奏可以留给后续 revise。

When to use:
- 当 lecture plan 已经基本确定，需要把筛过的材料真正写成一版连续草稿时。
- 当希望 Skill 4 主要做重写、删改和教学化 revise，而不是再回头大量翻参考资料时。

Artifact flow:
- In: lecture plan + related materials
- Out: rough lecture draft

Responsibilities:
- 按 lecture plan 的 section 顺序，直接写出连续草稿，而不是只写标题、要点或 section schema。
- 尽量保留参考材料中值得进入讲义的 insight、解释路径、例子、反例、桥接句和机制说明。
- 对每个 section 保留足够材料，使 Skill 4 默认无需再查原始参考文件也能完成重写。
- 对关键信息、精彩叙述候选和重要机制附 source anchors，保证后续可追溯。
- 允许局部冗余、重复、语气不统一，只要这些内容有助于保全材料并支持后续 revise。
- 必要时补充课件之外但对项目目标重要的知识点，但应与原始参考材料显式区分。
- 标注哪些位置在 Skill 4 中需要重点处理，例如顺序待重组、术语跳跃、解释太平、段落重复或桥接不足。

Non-goals:
- 不追求一次性写成最终讲义。
- 不为了流畅或简洁而过早删掉可能有价值的 insight、例子、机制解释或过渡表述。
- 不把粗草稿写成原文摘抄拼贴或大段复制；应整理成可继续重写的教程草稿。
- 不在当前阶段直接写最终 demo 代码；如果需要最终演示代码与插入，由 Skill 6 负责；如果需要从现成图片资源中挑选并插入图片，由 Skill 7 负责。

Inputs:
- 单个 lecture 的计划文档。
- 多份与该 lecture 相关的课程 PDF、讲义、论文、笔记或其他资料。
- 项目目标与写作约定。
- 必要时的补充搜索结果。

Outputs:
- 一版按 lecture plan 展开的连续粗草稿。
- 按 section 组织的材料保留结果，包括应保留的 insight、例子、机制解释和桥接信息。
- 关键事实、精彩叙述候选或重要机制的 source anchors。
- 交给 Skill 4 的 revise 提示，说明哪里需要重组、删冗余、补桥接或统一语气。

Suggested artifacts:
- `specs/lectures/<lecture_name>_draft.md`

Negative constraints:
- 不要只交标题加 bullet points。
- 不要把所有参考资料机械拼接成材料仓库。
- 不要为了“像成稿”而过早抹平参考资料中真正有价值的 insight 和讲法。
- 不要省略关键内容的来源锚点，否则 Skill 4 难以在必要时回查。

### 3. Skill: Lecture Drafting

Goal:
- 对整讲 rough drafting 做编排：将不同 section 分派给多个 `rough-drafter` subagent，由这些 subagent 执行 `rough-draft`，再由主 agent 统一合并成一版材料完整的 lecture 粗草稿。
- 该 skill 的核心任务是 lecture-level drafting orchestration 与 merge，而不是替代 `rough-draft` 去亲自重写每一个局部 section。

When to use:
- 当单个 lecture 的资料较多，适合多个 subagent 并行起草不同 section 时。
- 当 references 需要按 section 分配、并且开头、结尾、全局术语或 source anchor 风格应由一个统一 owner 负责时。

Artifact flow:
- In: lecture plan + references + optional existing draft
- Out: integrated rough lecture draft

Responsibilities:
- 读取 lecture plan、现有草稿和提供的 references，识别可并行 drafting 的 section 边界。
- 生成一份简短 drafting brief，明确 section ownership、reference allocation、术语/记号约束、source anchor 风格，以及哪些全局段落只允许主 agent 处理。
- 将不同 `##` 或必要时的 `###` section 分配给多个 `rough-drafter` subagent，并明确各自 write scope；这些 subagent 只执行 `rough-draft`。
- 要求 worker 只负责局部 section 的材料保全与粗草稿写作，不负责 lecture 开头/结尾、全局 framing 或整讲术语归一。
- 在 worker 返回后，由主 agent 统一合并各 section 粗稿，处理标题层级、明显重复、术语一致性、source anchor 风格，以及最小必要的 section 连接。
- 确保最终文稿读起来像一版完整 rough draft，而不是多个 section draft 的机械拼接。

Non-goals:
- 不用于单一 section 的局部起草；这类任务交给 `Skill 2: Rough Draft`。
- 不让每个 worker 都重读全部 references；应尽量按 section 分配最相关材料。
- 不在 merge 阶段把粗草稿提前 polish 成讲义；教学化重写交给 `Skill 4: Writing Revision`。
- 不最终决定 demo 或图片插入。

Inputs:
- 单个 lecture 的计划文档。
- 多份与该 lecture 相关的课程 PDF、讲义、论文、笔记或其他资料。
- 可选的现有 lecture 粗草稿。
- 项目目标与写作约定。

Outputs:
- 一版经过整讲合并后的 rough lecture draft。
- 一份简短 drafting brief。
- 对 section-level 分工与 merge 重点的简要说明。

Suggested artifacts:
- `docs/<target_lecture>.md`
- `specs/lectures/<lecture_name>_drafting_brief.md`

Negative constraints:
- 不要让多个 worker 同时改 lecture 开头、结尾或全局术语。
- 不要把整讲 merge 降级成“把多个 section 原样拼起来”。
- 不要为了做 merge cleanup 而过早删掉对后续 revision 仍有价值的 source anchors、例子或局部 handoff。

### 4. Skill: Writing Revision

Goal:
- 对 Skill 2 或 Skill 3 产出的粗草稿进行重写和 revise，使其在整体写作气质上达到“深入浅出、由具体到抽象、由问题到机制”的要求。
- 该 skill 的核心任务是重组、删冗余、统一语气和打通认知路径，而不是再次做一轮大规模资料抽取。

When to use:
- 当 lecture 已经有一版材料比较全的粗草稿，但讲解不够顺、太像材料堆叠、或不够“深入浅出”时。
- 当需要把粗草稿改写成更接近最终讲义的教学叙述时。

Artifact flow:
- In: rough lecture draft
- Out: revised draft

Responsibilities:
- 检查内容是否沿着读者的认知路径推进，而不是沿着作者的知识结构堆叠。
- 调整讲解顺序，使内容尽量遵循“问题或现象 -> 机制直觉 -> 必要抽象 -> 验证或应用”的默认推进方向。
- 合并重复材料，删除只回答“是什么”却没有回答“为什么需要”或“解决了什么问题”的段落。
- 将抽象内容尽量绑定到最小例子、可运行代码、图示或反例中的至少一种。
- 保留粗草稿中已经保留下来的高价值 insight、好例子和好桥接，不要在重写过程中把它们误删或抹平。
- 默认基于粗草稿完成 revise；只有当来源冲突、证据不足或内容明显缺失时，才回原始参考文件核对。
- 在不牺牲严谨性的前提下，提升文字的可读性、推进感和教学性。
- 可以指出某处“需要一个例子或演示”，但不负责最终决定是否插入 code block，也不负责生成最终代码块。

Non-goals:
- 不机械模仿某一教材的句式、段落模板或表面文风。
- 不强制每个小节使用完全一致的结构。
- 不默认重新通读全部参考资料；粗草稿应当是主要工作底稿。
- 不把 revise 简化成只修语气和措辞；重点仍然是认知路径与教学推进。
- 不作为最后一步去直接写入或删除具体 demo 代码块；demo 的最终生成与插入由 Skill 6 负责，现成图片资源的筛选与插入由 Skill 7 负责。

Inputs:
- Skill 2 或 Skill 3 产出的 lecture 粗草稿。
- 项目目标、读者画像与写作约定。
- 只有在必要时才使用的原始课件、补充材料或相关代码。

Outputs:
- 修订后的章节或 lecture 草稿。
- 对主要重组与改写点的简要说明。
- 对仍然存在的理解断层、术语跳跃、示例不足或证据缺口的提示。

Suggested artifacts:
- `docs/<target_lecture>.md`
- `specs/lectures/<lecture_name>_revision_notes.md`

Instruction template:
- 写作应以认知推进为核心，而不是以知识罗列为核心。
- 把 section 目标当作内部写作脚手架，而不是正文中的显式标签；正文默认不写 `本节目标`、`这一节要说明的是`、`这一段真正想说的是` 这类元话语。
- 优先回答“为什么会有这个问题”与“这个机制解决了什么”，再回答“它的正式定义是什么”。
- 讲解顺序应尽量遵循：问题或现象 -> 机制直觉 -> 必要抽象 -> 验证或应用。
- 这不是固定模板；当内容不适合完整走这条路径时，也应至少保证读者知道：当前在解决什么问题、为什么需要这个概念、以及这个概念如何落到代码、实验或系统现象上。
- 抽象概念的讲解，必须尽量绑定到最小例子、空的 code block placeholder、picture placeholder、图示说明或反例中的至少一种。
- 正式定义、公式和术语应在读者已经形成初步直觉后引入。
- 一次只引入当前问题所需的最小新概念，避免在单个小段中叠加过多新抽象。
- 如果当前 section 需要代码或图来完成理解闭环，但 Skill 6 或 Skill 7 尚未介入，可以显式留下空的 code block placeholder 或 picture placeholder，说明这里要观察什么、为什么需要它。
- 代码块必须服务于理解，不只是展示实现。
- 每一段文字都应当有推进性，让读者比上一段多看清一层机制。
- 允许不同章节采用不同展开方式，但整体上应保持“由具体到抽象、由现象到机制、由局部到系统”的叙述方向。
- 文档中的预期口吻应站在 tutorial 作者的角度，用“我们/下面来看/这里可以看到”这类作者叙述，而不是站在辅助 agent 角度写“这里需要补充”“后续可考虑”“建议模型”等元注释式表述；如果确实需要交接信息，应单独放在注释或 revision notes 中。
- 正文默认应把讲义写成自足文本，不要提“参考材料里”“课件这里”“slides 上”这类来源感知表述；如果某个比喻或机制解释值得保留，就直接写成讲义自己的叙述。

Revision acceptance checklist:
- 作者在内部检查时，必须能用一句话回答当前 section 在解决什么问题；但这个检查默认不应直接暴露成正文中的 `本节目标` 标签。
- 关键抽象概念在首次正式定义前，已经出现问题感、现象、例子、反例或直觉入口中的至少一种。
- section 与 section 之间存在显式桥接，而不是突然切换话题。
- revise 后保留了粗草稿中真正高价值的 insight、例子和桥接，没有为了统一语气而把它们抹平。
- 如果某处理解闭环依赖代码或图，但当前仍未实现，正文中已有明确的 code block placeholder 或 picture placeholder。
- 正文语气站在 tutorial 作者角度，而不是辅助 agent、编辑者或审稿者角度。

Negative constraints:
- 不要从定义列表开始。
- 不要为了完整而过早展开旁支内容。
- 不要把名词解释写成百科词条。
- 不要把 code block 当作展示实现能力的堆砌。
- 不要默认读者会自动把数学、代码和系统现象联系起来；需要显式搭桥。
- 不要让一节内容只回答“是什么”，却不回答“为什么重要”。
- 不要把正文写成面向协作者的编辑备忘录；面向作者协作的信息应放进注释或单独的 revision notes。
- 不要把正文写成作者的写作脚手架展示；避免 `本节目标`、`这一节真正想说明的是`、`下面我们将证明` 这类结构暴露语。
- 不要在正文里提参考材料、课件或 source 文档的存在；读者默认不需要知道这些来源。
### 5. Skill: Lecture Revision

Goal:
- 对整讲 lecture revision 做编排：将不同 section 分派给多个 `writing-reviser` subagent，由这些 subagent 执行 `writing-revision`，再由主 agent 统一完成整讲收口，并在最后执行一次 audit-only 二次检查。
- 该 skill 的核心任务是 lecture-level orchestration、integration 与窄范围 residue audit，而不是替代 `writing-revision` 去重写每一个局部 section。

When to use:
- 当用户希望整讲 revision 可以并行分配给多个 subagent 时。
- 当上一讲顺承、全讲术语一致、section 间桥接和最终去重复应由一个统一 owner 负责时。

Artifact flow:
- In: rough lecture draft or section-revised lecture draft
- Out: integrated revised lecture draft

Responsibilities:
- 读取整讲草稿，识别可并行 revision 的 section 边界。
- 默认将同文件夹内编号减一的 lecture 视为上一讲，只由主 agent 生成一份简短的 continuity brief。
- 将不同 `##` 或必要时的 `###` section 分配给多个 `writing-reviser` subagent，并明确各自 write scope；这些 subagent 只执行 `writing-revision`。
- 要求 worker 只负责局部 section 重写，不负责上一讲顺承、全讲 framing 或全局术语改写。
- 在 worker 返回后，由主 agent 统一处理上一讲到当前讲开头的承接、section 间桥接、术语和记号一致性，以及重复内容删除。
- 在整讲 integration 之后，执行一次 audit-only 二次检查；该检查只负责拦截写作脚手架、课程结构元话语、source-aware 残留、术语/记号不一致、公式转义污染和明显重复 framing 等 lecture-level 残留问题。
- 确保最终文稿读起来像一讲完整讲义，而不是多个 section revised draft 的拼接。

Non-goals:
- 不用于单一 section 的局部重写；这类任务交给 `Skill 4: Writing Revision`。
- 不让每个 worker 都各自负责上一讲顺承。
- 不在 integration 阶段重写全部 section 的局部 prose；主 agent 应主要负责收口而不是重复 worker 的工作。
- 不把 audit-only 二次检查升级成第二轮完整 revision；若问题需要大段重写，应显式记录而不是静默重写。
- 不最终决定 demo 或图片插入。

Inputs:
- 单个 lecture 的粗草稿或已部分 revised 的讲义。
- 项目目标、读者画像与写作约定。
- 默认需要时读取的上一讲：同文件夹内编号减一的 lecture。

Outputs:
- 一版经过整讲集成后的 revised lecture draft。
- 一份简短 continuity brief。
- 对 section-level 分工与整讲收口点的简要说明。

Suggested artifacts:
- `docs/<target_lecture>.md`
- `specs/lectures/<lecture_name>_continuity_brief.md`

Negative constraints:
- 不要让多个 worker 同时修改 lecture 开头、结尾或全局术语。
- 不要为了制造顺承而重复上一讲已经讲清的大段内容。
- 不要把整讲 integration 降级成“把多个 section 原样拼起来”。

### 6. Skill: Demo Insertion

Goal:
- 在单个 lecture 中识别哪些位置真正需要 demo，并在这些位置直接生成并插入极简、可运行的 code block。
- 当 demo 的教学收益明显高于纯文字说明时，不只给出建议，而是把 demo 本身写进 lecture；必要时也可以插入由这些代码直接生成的最小观测图。
- 该 skill 是需求驱动的：先从讲义哪里存在理解缺口出发，再决定是否需要 demo，而不是先看手头有什么素材可插。

When to use:
- 当 lecture 的文字结构和写作逻辑已经基本稳定，需要把真正的教学 demo 落到文稿里时。
- 当某些 section 只有通过运行一个极小实验、观察一条曲线或比较两种行为，读者才容易真正建立直觉时。

Artifact flow:
- In: revised lecture draft
- Out: lecture draft with inserted runnable demos

Responsibilities:
- 判断某一段概念是否值得通过 demo 来建立直觉、展示现象、验证机制或比较行为。
- 只在 demo 能明显提升理解时插入 code block；如果文字、图示或公式已经足够清楚，则不插入。
- 为适合演示的概念设计最小可运行代码块，并把它们直接写入 lecture，优先展示机制而不是完整系统。
- 当最小 demo 的自然输出就是一张曲线、散点图或示意图时，可以把这类由 demo 直接生成的图作为 demo 的一部分插入。
- 让 demo 与前后的 text block 紧密配合，使读者知道代码在观察什么、为什么值得运行，以及运行后应看到什么。
- 默认少写 demo；只有在 demo 能显著增加理解时才写。

Non-goals:
- 不为了“看起来像 notebook”而机械增加代码块数量。
- 不写工程化、生产化或可复用性优先的实现。
- 不引入防御性编程、配置系统、日志系统、命令行接口、类层次或多余抽象。
- 不把 tutorial code block 写成完整库函数或项目模块。
- 不从网页或现成图片库里大量搜图；现成图片资源的筛选与插入由 Skill 7 负责。
- 不扩展 lecture 的理论覆盖范围。
- 不只停留在“建议哪里可以加 demo”；如果决定需要 demo，就应直接生成并插入。

Decision rules:
- 当代码能够帮助读者观察一个现象、理解一个机制、比较两种行为、或验证一个推导时，优先考虑插入 demo。
- 当内容主要是定义、历史背景、术语整理、经验总结或高层框架时，通常不插入 demo。
- 当某段内容若没有运行一个极小例子就会显得空泛，或者读者运行后能立刻获得直觉时，应考虑插入 demo。
- 如果 demo 不能明显带来新的观察，而只是把文字重复翻译成实现，则不要插入。
- 如果某个理解缺口更适合由现成图片来弥补，而不是由代码运行结果来弥补，应把任务交给 Skill 7，而不是勉强写 demo。

Code style constraints:
- 代码必须极简，像能运行的伪代码。
- 优先使用若干个短函数，而不是一个大而全的脚本。
- 只保留理解当前机制所需的最小参数、最小数据和最小控制流。
- 不写防御性分支、边界检查、异常处理、日志、配置对象、命令行入口、测试脚手架或性能优化样板。
- 不为了“规范”而增加类封装、抽象基类、工厂模式或其他冗余设计。
- 变量名、函数名和输入都应服务于教学清晰度，而不是工程通用性。

Inputs:
- 单个 lecture 的 revised draft。
- 当前 lecture 的 section 目标与上下文。
- 必要时的相关资料、参考实现或已有 placeholder。

Outputs:
- 已直接插入 demo 后的 lecture 文稿。
- 每个 demo 的教学目的说明。
- 极简、可运行的最终 code block。
- 若 demo 会自然产出图形，则给出相应的最小生成图与插入方式。
- 对未插入 demo 的关键位置给出简短原因。

Suggested artifacts:
- `docs/<target_lecture>.md`
- `specs/lectures/<lecture_name>_demos.md`

### 7. Skill: Figure Curation

Goal:
- 在单个 lecture 中读取现成图片资源，筛选哪些图真正值得进入讲义，以及它们应插在什么位置。
- 该 skill 是资源驱动的：先看已经有哪些图可用，再判断其中哪些图有教学价值，而不是先假设某处一定要插图。
- 支持两类图片来源：仓库内已有图片（例如 `@image/<part name>` 下的图片）以及用户提供网页中的现成图片链接。

When to use:
- 当 lecture 的文字结构和写作逻辑已经基本稳定，并且用户已经提供一批本地图片或网页链接时。
- 当我们希望复用现成图片来帮助读者建立几何直觉、趋势判断、结构印象或实验现象时。

Artifact flow:
- In: revised lecture draft + candidate figure resources
- Out: figure curation plan / inserted image blocks

Responsibilities:
- 读取现有图片资源，判断哪些图真正有教学价值，哪些图虽然存在但不值得插入。
- 将选中的图片映射到具体 section 或段落，说明它为什么放在这里、希望读者从图里看到什么。
- 对本地图片，插入指向仓库内已有图片的 image block。
- 对网页图片，在用户已经提供网页的前提下，定位适合插入的具体图片 URL，并插入指向该图片 URL 的 image block。
- 为每张保留的图补最小必要的 caption、引导语或观察提示，使图片与上下文叙述连起来。
- 默认宁缺毋滥；如果一张图不能明显改善理解，即使它可用，也不应插入。

Non-goals:
- 不生成新的图片、曲线或示意图；需要代码生成的图交给 Skill 6。
- 不从零开始做大规模开放式搜图；默认只处理用户给定的本地图资源或网页。
- 不因为“手头有图”就强行往讲义里插图。
- 不扩展 lecture 的理论覆盖范围。

Decision rules:
- 当一张现成图片能更快帮助读者建立几何直觉、趋势判断、结构印象或实验现象时，应考虑插入。
- 当图片只是重复正文已经足够清楚的内容，或者需要很长解释才能看懂时，通常不插入。
- 当多张图表达的是同一个意思时，优先保留最直接、最干净、与当前 section 最贴合的一张。
- 当网页中有多张候选图时，应优先选择来源稳定、主题聚焦、与正文叙述粒度一致的图片。

Inputs:
- 单个 lecture 的 revised draft。
- 用户提供的本地图片资源，例如 `@image/<part name>`。
- 用户提供的网页链接；必要时从网页中定位具体图片 URL。
- 当前 lecture 的 section 目标与上下文。

Outputs:
- 候选图片资源的筛选结果。
- 建议插入图片的位置清单。
- 每张保留图片的教学目的、caption 或引导语。
- 指向本地图片路径或具体网页图片 URL 的 image block。
- 不建议插入的图片及原因。

Suggested artifacts:
- `specs/lectures/<lecture_name>_figures.md`
- `specs/lectures/<lecture_name>_image_blocks.md`

### Workflow Boundary

- Skill 1 负责单个 lecture 的资料筛选、内容取舍与 section 结构设计。
- Skill 2 负责在既定 lecture plan 下写出一版材料完整的 lecture 粗草稿。
- Skill 3 负责整讲级的 rough-draft orchestration：可并行分派多个 `rough-drafter` subagent，由它们执行 `Skill 2`，并在最后统一合并成一版材料完整的 lecture 粗草稿。
- Skill 4 负责单 section 或单 lecture局部范围内的重写、删改和教学风格上的 revise。
- Skill 5 负责整讲级的 revision orchestration：可并行分派多个 `Skill 4` worker，并在最后统一处理上一讲顺承、section 桥接、术语一致性、去重复，以及一次 audit-only 二次检查。
- Skill 6 负责在最后阶段按理解缺口判断是否需要 demo，并把极简、可运行的 tutorial demos 直接生成并插入 lecture。
- Skill 7 负责在最后阶段基于现成图片资源做 figure curation：先看有哪些图可用，再决定哪些图值得插入以及插在哪里。
- Skill 1 先出方案，再通过用户交互收敛，不默认一次性定稿。
- Skill 1 不写教程正文。
- Skill 2 不重做 lecture 层面的内容取舍，除非发现明显缺口；但它必须把应保留的材料真正写进粗草稿，而不是只给提纲。
- Skill 2 不负责 lecture-level 的并行 drafting orchestration 与整讲 merge；这些任务由 Skill 3 统一负责。
- Skill 3 负责唯一一次 lecture-level drafting brief，并确保 references 分配、全局术语和 source anchor 规则只在整讲层面统一处理，不由各 section worker 重复承担。
- Skill 3 只做 rough-draft 阶段所需的最小 merge cleanup，不把整讲提前 polish 成讲义。
- Skill 4 默认基于 Skill 2 或 Skill 3 产出的粗草稿完成局部 revise，不再重新通读参考资料；只有在来源冲突、证据不足或材料明显缺失时才回源核对。
- Skill 4 不负责 lecture-level 的上一讲顺承与整讲 integration；这些任务由 Skill 5 统一负责。
- Skill 4 可以为保证局部认知连续性少量补写桥接段、最小例子、空的 code block placeholder 或 picture placeholder，但不应借此扩展 lecture 的理论覆盖范围，也不应重做大规模资料抽取。
- Skill 5 负责唯一一次 lecture-level continuity brief，并确保上一讲顺承只在整讲层面处理，不由各 section worker 重复承担。
- Skill 5 可以建议哪里需要例子、demo 或图片，但不最终生成并插入 demo code block，也不负责现成图片资源的最终筛选。
- Skill 5 的 audit-only pass 只允许窄范围、低风险的 lecture-level 清理；如果问题需要第二轮完整重写，应明确暴露为后续任务，而不是在 audit 中静默完成。
- Skill 6 不负责扩展 lecture 的理论覆盖范围；它只在必要位置直接生成并插入极简、可运行的 demo。
- Skill 7 不负责扩展 lecture 的理论覆盖范围；它只基于已有图片资源判断哪些图值得插入，以及这些图应如何与正文配合。
