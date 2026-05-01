# HANDOVER

## 当前目标

从技术人员和产品经理两个视角完整审阅 CourseRepoFinder，并围绕南开大学计算机科学与技术 2024 专业培养计划、NKUCS.ICU 课程经验站和 NKUCS.ICU GitHub Issues 改进项目。

参考入口：

- https://nkucs.icu/#/
- https://github.com/NKUCS-ICU/NKUCS.ICU/issues

## 项目理解

CourseRepoFinder 是一个面向高校课程作业仓库的 GitHub 检索与推荐系统。主链路是：

1. `app.py` 接收用户输入并展示查询解析、检索计划和结果。
2. `query_parser.py` 识别学校、课程、作业类型、编号、语言和技术栈。
3. `search_planner.py` 根据意图生成多路 GitHub repository search 查询。
4. `github_client.py` 拉取仓库、README、用户公开仓库，并由 `cache_store.py` 做缓存。
5. `rrf.py` 做多路召回融合。
6. `evidence_builder.py` 从 repo name、description、topics、README、owner profile 中抽取证据。
7. `hybrid_ranker.py` 综合课程、学校、作业、owner、质量、新鲜度、RRF 分数排序。

## 技术视角不足

- 依赖声明缺失：代码依赖 `yaml`，但 `requirements.txt` 和 `pixi.toml` 原本没有声明 PyYAML。
- 课程知识库过窄：原 `courses.yaml` 只覆盖少量核心课程，无法支撑培养计划中的大量专业必修和专业选修课。
- 短英文别名误匹配：`AI` 会命中 `nankai`，`C` 会命中 `computer`，这会污染课程解析、技术栈识别和排序证据。
- 别名重复会生成重复检索任务：例如 canonical name 和 aliases 同时包含同一课程名。
- 旧模块文档未同步：README 仍指向 `course_aliases.py`、`github_searcher.py`、`relevance_scorer.py` 作为主模块，但当前主链路已经迁移到 KB + planner + client + ranker。
- 评测薄弱：只有模块 smoke test，没有基于真实查询和标注结果的排序质量评测。
- GitHub API 错误可见性不足：UI 只显示没有结果，用户不容易区分“无结果”“限流”“网络失败”。

## 产品视角不足

- 用户不知道系统为什么这么搜：原 UI 展示结果和证据，但没有展示检索计划。
- 课程覆盖不透明：用户无法判断某门课是否在知识库中。
- 对南开场景的定位不够强：没有直接链接到 NKUCS.ICU 和课程讨论入口。
- 搜索快捷入口较少：只覆盖部分大二大三核心课，信息检索、机器学习等常见检索需求不突出。
- 风险提示仍偏工程化：例如“学校信号来自作者其他仓库”有价值，但还可以进一步转成普通用户更容易理解的可信度说明。

## 本轮已完成改进

- 2026-05-01 追加修复：
  - 修复 Streamlit 搜索时 `AttributeError: 'NoneType' object has no attribute 'lower'`。
  - `matching.contains_signal()` 现在接受任意对象输入，遇到 `None` 会安全返回 `False`。
  - `evidence_builder.py` 对 GitHub 返回的空 `full_name/name/description/readme_text/topics` 做了清洗。
  - 增加空值安全回归测试。
- 2026-05-01 追加合集仓库支持：
  - 在 `schools.yaml` 中维护南开课程合集种子仓库和种子用户。
  - 为 `Absurdaaa/NKU_CS_courses`、`TephrocactusHC/NKUCS-SAVE`、`Starlight0798/NKU-share`、`fscdc/NKU-CS-HELP` 维护本地课程目录提示。
  - `github_client.py` 增加 code/path search 和目录树获取。
  - `search_planner.py` 为课程查询生成 `code_path` 检索任务。
  - `search_engine.py` 把 code search 的文件命中归并到仓库，并在搜索时注入种子合集。
  - `evidence_builder.py` 将目录/文件路径作为课程证据，推荐理由会出现“目录/文件路径命中课程关键词”。
  - `rrf.py` 修正来源加成，让 `exact_*`、`code_path_*`、`seed_collections` 能正确获得召回路线 bonus。
- 2026-05-01 追加通用 Query Expansion：
  - 新增 `query_expander.py`，把用户自然输入扩展为有优先级、有 route、有预算控制的 GitHub Search query。
  - 支持学校别名、课程别名、资源类型词、词序变化、英文表达、GitHub 命名习惯和 `in:path` 路径检索。
  - LLM 扩展为可选能力，通过 `ENABLE_LLM_QUERY_EXPANSION=true` 启用，从 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` 读取配置，不硬编码密钥。
  - `query_parser.py` 增加 `resource_type` / `resource_aliases`，可识别“资料、笔记、课件、试卷、复习、notes、slides、exam”等资源意图。
  - `courses.yaml` 增加 `advanced_mathematics`，但 Query Expansion 是泛化模块，并非只服务高数。
  - `app.py` 的调试面板现在展示学校别名、课程别名、资源词和实际搜索计划。
  - `.env.example` 已改为占位符，`.gitignore` 增加 `.env`、`.cache/`、`__pycache__/`。
- 新增 `matching.py`，统一处理文本信号匹配。
  - 中文和长英文短语仍支持子串命中。
  - `AI`、`OS`、`DB`、`IR`、`C`、课程代码等短 ASCII 信号必须按 token 边界命中。
- 修复查询解析：
  - `lab2`、`hw1`、`实验3` 现在能同时识别作业类型和编号。
  - `Nankai computer network` 不再被误识别为 C 语言。
  - `nankai` 不再触发人工智能课程。
- 扩充 `courses.yaml`：
  - 加入南开计科 2024 培养方案中的大类基础、专业必修、专业选修、国际学分课程等关键课程。
  - 增加课程代码、中英文别名、常见技术栈和作业关键词。
  - 结合 NKUCS.ICU Issues 中出现的课程代码变体，例如 `密码学基础` 同时保留 `CSSE0059` 与 `CSSE0047`。
- 更新 UI：
  - 侧边栏显示课程知识库覆盖数量。
  - 增加 NKUCS.ICU 与 GitHub Issues 快捷入口。
  - 增加检索计划预览，展示 route 与实际 GitHub 查询语句。
  - 快捷搜索加入信息检索系统原理、机器学习及应用。
- 更新 README：
  - 修正主模块说明。
  - 增加 Pixi 使用方式、课程来源、当前局限和扩展方向。
- 补充依赖：
  - `requirements.txt` 增加 `PyYAML>=6.0.0`。
  - `pixi.toml` 增加 `pyyaml>=6.0.0`。
- 更新测试：
  - 增加短别名误匹配回归测试。
  - 增加 `lab2` 类型识别测试。

## 已验证

运行：

```bash
python test_modules.py
```

结果：所有模块测试通过。

## 建议下一步

1. 增加真实查询评测集，例如每门课维护 3-5 个正例仓库和若干负例仓库。
2. 在 `SearchEngine._execute_plan` 中保留每路 API 错误，UI 展示限流/网络失败/无结果的区别。
3. 为 `courses.yaml` 增加课程类别、学分、建议学期字段，并在 UI 中支持按培养计划浏览课程。
4. 引入轻量语义召回或 README 摘要，弥补关键词命中不到但语义相关的仓库。
5. 清理或明确 legacy 模块：`course_aliases.py`、`github_searcher.py`、`relevance_scorer.py` 当前不是主链路，后续可删除或标记为兼容层。
