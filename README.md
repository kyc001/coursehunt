# CourseHunt

**Hunt the right repo for your course**

面向高校课程作业的 GitHub 仓库智能检索系统

## 解决的问题

GitHub 原生搜索不理解 "南开大学 并行程序设计 lab2" 的语义关系。CourseHunt 通过课程知识库、学校识别、作业特征分析和用户画像，帮你精准定位同校课程仓库。

## 核心功能

- **课程知识库**：内置南开大学 2024 培养方案核心课程，支持中英文别名、课程代码
- **查询理解**：自动识别学校、课程、作业类型、作业编号
- **多路召回**：生成多种搜索策略，提高召回率
- **RRF 融合**：多路结果融合排序，避免单一查询主导
- **用户画像**：通过作者其他仓库推断学校背景
- **可解释推荐**：展示推荐理由和命中证据

## 快速开始

### Streamlit Cloud 部署

1. Fork 本仓库
2. 访问 [share.streamlit.io](https://share.streamlit.io)
3. 选择 fork 的仓库，设置主文件为 `app.py`
4. 在 Secrets 中添加环境变量：

```toml
GITHUB_TOKEN = "your_github_token"
EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_API_KEY = "your_embedding_key"
EMBEDDING_MODEL = "BAAI/bge-m3"
OPENAI_API_KEY = "your_openai_key"
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_MODEL = "openai/gpt-oss-120b:free"
ENABLE_LLM_QUERY_EXPANSION = "true"
```

### 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API keys

# 运行
streamlit run app.py
```

## 系统架构

```
用户输入查询
    ↓
查询理解 (学校/课程/作业识别)
    ↓
查询扩展 (知识库 + 可选 LLM)
    ↓
多路召回 (GitHub API)
    ↓
RRF 融合 + 证据抽取
    ↓
混合排序 (动态权重)
    ↓
可解释推荐结果
```

## 支持的课程

当前重点维护南开大学计算机专业课程：

| 课程 | 课程代码 |
|------|---------|
| 数据结构 | COSC0007 |
| 离散数学 | COSC0050 |
| 算法导论 | COSC0016 |
| 并行程序设计 | COSC0025 |
| 操作系统 | COSC0009 |
| 编译系统原理 | COSC0017 |
| 数据库系统 | COSC0013 |
| 计算机网络 | COSC0010 |
| 软件工程 | COSC0048 |
| 信息检索系统原理 | COSC0032 |
| 机器学习及应用 | COSC0028 |
| 深度学习及应用 | COSC0054 |

课程来源：[南开大学计算机学院 2024 培养方案](https://nkucs.icu/#/)

## 技术栈

- **前端**：Streamlit
- **搜索**：GitHub REST API
- **排序**：RRF + 多维度加权
- **缓存**：SQLite + 内存缓存
- **LLM**：OpenRouter (可选)

## License

MIT
