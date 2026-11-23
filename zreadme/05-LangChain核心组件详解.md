# 05 - LangChain 核心组件详解

> 落地代码：`internal/handler/app_handler.py` 的 `debug()`
> 前置阅读：`04-后端架构与请求链路.md`
> 本篇为**总览 + 索引**，各模块已拆成独立文档，每篇末尾都附「不用 LangChain 时怎么写」的对照实现。

## 读完你能做什么

- 说明 LangChain 解决什么问题，`langchain-core` / `langchain-openai` / `langchain-community` 各管什么；
- 熟练使用消息类型、Prompt 模板、ChatModel、OutputParser；
- 理解 `prompt | llm | parser` 的原理（`Runnable.__or__`）与每一环的数据类型；
- 用 `RunnableParallel` / `RunnablePassthrough.assign` 写出 RAG 最小雏形；
- 用回调观察链的内部过程，知道 LangSmith 怎么开；
- 逐行讲清项目里那 5 行核心代码，并避开高频坑；
- 说得清每个组件在**不用 LangChain** 时对应哪段手写代码，以及手写版缺了什么。

---

## 模块索引

| 文档 | 对应模块 | 手写对照内容 |
| --- | --- | --- |
| [05-1-消息体系.md](./05-1-消息体系.md) | `SystemMessage` / `HumanMessage` / `AIMessage` / `ToolMessage` | `messages` 就是 `list[dict]`，`role` 字符串手写 |
| [05-2-Prompt组件.md](./05-2-Prompt组件.md) | `PromptTemplate` / `ChatPromptTemplate` / `MessagesPlaceholder` / `partial` / 拼接 / 复用 | f-string、`str.format`、`functools.partial` |
| [05-3-Model组件.md](./05-3-Model组件.md) | `ChatOpenAI` / `invoke` / `batch` / `stream` / 环境变量 | `client.chat.completions.create`、线程池、`stream=True` |
| [05-4-OutputParser.md](./05-4-OutputParser.md) | `StrOutputParser` / `JsonOutputParser` | 抠 `choices[0].message.content`、手写格式说明 + `json.loads` |
| [05-5-LCEL与Runnable.md](./05-5-LCEL与Runnable.md) | LCEL、`RunnableSequence` / `RunnableParallel` / `RunnablePassthrough` | for 循环串联、`ThreadPoolExecutor` 并行、手写 RAG |
| [05-6-回调与可观测.md](./05-6-回调与可观测.md) | `BaseCallbackHandler` / `StdOutCallbackHandler` / LangSmith | 每个调用点手写日志与计时 |
| [05-7-落地与换厂商.md](./05-7-落地与换厂商.md) | `debug()` 逐行拆解、换模型类 | 手写等价 `debug()`、换厂商要改哪些地方 |

---

## 一、LangChain 是什么

直接调模型时会反复写「拼字符串 → 组 messages → create → 抠 `choices[0].message.content`」，换模型、加历史、做流式、做批量时又要重写一遍。这些与业务无关、但每个项目都重复的代码，就是 LangChain 要抽象的：

| 问题 | 答案 |
| --- | --- |
| 提示词怎么组织复用 | Prompt 组件 |
| 不同厂商怎么统一调用 | ChatModel 抽象（`invoke`/`batch`/`stream`） |
| 输出怎么变结构化 | OutputParser |
| 多步骤怎么串 | LCEL + Runnable |
| 过程怎么观测 | Callbacks / LangSmith |

### 不用 LangChain 时：同一件事要手写一遍

上面说的「拼字符串 → 组 messages → create → 抠 `choices[0].message.content`」，展开就是下面这样
（项目已安装 `openai` SDK，`app_handler.py` 里也 import 了它）：

```python
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),  # 兼容接口一般以 /v1 结尾
)


def chat(query: str, history: list[dict] | None = None) -> str:
    # 1. 拼字符串 + 组 messages：system 人设、历史消息、本轮提问全靠手写
    messages = [{"role": "system", "content": "你是一个乐于助人的助手"}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"请回答：{query}"})

    # 2. create：模型名、参数都直接写死在调用里
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        temperature=0.7,
    )

    # 3. 抠 choices[0].message.content，还要自己维护多轮历史
    answer = resp.choices[0].message.content or ""
    if history is not None:
        history.append({"role": "user", "content": f"请回答：{query}"})
        history.append({"role": "assistant", "content": answer})
    return answer
```

同样的效果，用 LangChain 写：

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手"),
    MessagesPlaceholder("chat_history"),
    ("human", "{query}"),
])
chain = prompt | ChatOpenAI(model="deepseek-flash", temperature=0.7) | StrOutputParser()

answer = chain.invoke({"query": query, "chat_history": history or []})
```

两种写法逐项对比：

| 环节 | 手写 openai SDK | LangChain |
| --- | --- | --- |
| 拼提示词 | f-string / 字符串拼接，变量多了容易漏 | `ChatPromptTemplate` 用 `{变量}` 声明，缺变量直接 `KeyError` |
| 组 messages | 手写 role/content 字典，role 拼错不报错 | `("system", ...)`、`("human", "{query}")` 由框架转换 |
| 调模型 | 参数名随厂商/版本变化 | `ChatOpenAI(model=...)` 统一封装，`invoke`/`batch`/`stream` 一致 |
| 取结果 | 自己写 `resp.choices[0].message.content` 并判空 | `StrOutputParser()` 直接产出 `str` |
| 换模型 | 换 SDK、改参数、重写结果解析 | 只换模型类，链不动 |
| 流式 | 手动遍历 chunk、拼接 `delta.content` | `chain.stream(...)` 直接产出增量字符串 |
| 批量 | 自己写循环或并发 | `chain.batch([...])` |
| 可观测 | 每步自己打日志 | Callback / LangSmith 自动串联 |

两点补充：

1. 「维护多轮历史」LangChain 默认**也不会**自动做，仍然要传 `chat_history`（或用 Memory 组件）；它省掉的是模板渲染、消息组装、调用参数、结果解析这些重复劳动。
2. 手写版把 `model`、`temperature` 写死在函数里，换模型要改代码；LangChain 版把「提示词」「模型」「解析器」拆成三个可替换的 Runnable，用 `|` 组装——这正是 LCEL 一节（[05-5-LCEL与Runnable.md](./05-5-LCEL与Runnable.md)）的基础。

### 三个包的分工

| 包 | 内容 | 本项目 |
| --- | --- | --- |
| `langchain-core` | 抽象与协议：`Runnable`、消息、Prompt、Parser、Callback、LCEL | 必需 |
| `langchain-openai` | OpenAI 兼容模型接入：`ChatOpenAI` | 必需 |
| `langchain` | 高层能力：Agent、工具编排 | 用不到 |
| `langchain-community` | 其他厂商/社区集成：千帆、Ollama、向量库 | 按需 |

`langchain`（元包）几乎不含核心代码，只是聚合。**学 LangChain 核心 = 学 `langchain-core`。** 项目实际 import：

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
```

> 本地 `requirements.txt` 未列 `langchain-core`、`langchain-openai`、`flask-cors`，但代码在用。补装：`pip install langchain-core langchain-openai flask-cors`。

---

## 二、组件速查表

| 环节 | 组件 | 输入 | 输出 | 详见 |
| --- | --- | --- | --- | --- |
| 消息 | `SystemMessage`/`HumanMessage`/`AIMessage` | —— | 消息对象 | [05-1](./05-1-消息体系.md) |
| 提示词 | `PromptTemplate`/`ChatPromptTemplate` | `dict` | `PromptValue` | [05-2](./05-2-Prompt组件.md) |
| 模型 | `ChatOpenAI` | `PromptValue`/消息列表 | `AIMessage`/`AIMessageChunk` | [05-3](./05-3-Model组件.md) |
| 解析 | `StrOutputParser`/`JsonOutputParser` | `AIMessage` | `str`/`dict` | [05-4](./05-4-OutputParser.md) |
| 编排 | `RunnableSequence`（`\|`） | 任意 | 任意 | [05-5](./05-5-LCEL与Runnable.md) |
| 并行/透传 | `RunnableParallel` / `RunnablePassthrough.assign` | `dict` | `dict` | [05-5](./05-5-LCEL与Runnable.md) |
| 观测 | `BaseCallbackHandler` | 事件 | —— | [05-6](./05-6-回调与可观测.md) |

---

## 三、常见坑

**坑 1：模板变量名与 invoke 键不一致**。模板写 `{question}`、调用传 `{"query": ...}` 就会 `KeyError`。排查：`print(prompt.input_variables)`。

**坑 2：忘记 `dotenv.load_dotenv()`**。`import dotenv` 不读文件，必须调用 `load_dotenv()`，且要在 `ChatOpenAI` **之前**，否则构造时报 Missing credentials。Flask 项目里 `app/http/app.py` 已调用过一次；单独写脚本调试时要自己调。

**坑 3：`OPENAI_API_BASE` 与 `OPENAI_BASE_URL` 混淆**。只配了一个、而代码读的是另一个，表现是「请求仍打到 `api.openai.com`，401 或超时」。生产代码建议显式传参。

**坑 4：把 stream 的 chunk 当完整消息**。`chain.stream(...)` 返回迭代器，`resp.content` 会 `AttributeError`；要 `for chunk in ...` 逐块拼。另一个隐蔽版本：循环里 `return chunk`，只返回第一块——流式接口必须用 `yield` 或 SSE。

**坑 5：JsonOutputParser 忘注入 format_instructions**

```python
prompt = ChatPromptTemplate.from_template("回答：{query}")   # 模型不知道要输出 JSON
```

正确写法：

```python
prompt = ChatPromptTemplate.from_template(
    "请根据用户的提问进行回答。\n{format_instructions}\n{query}"
).partial(format_instructions=parser.get_format_instructions())
```

**附加**：`langchain_core.pydantic_v1` 已废弃，改为 `from pydantic import BaseModel, Field`。

---

## 四、自测题（8 道）

1. 三个包各负责什么？只装 `langchain-core` + `langchain-openai` 能否跑通 `debug()`？
2. `prompt.format()`、`invoke().to_string()`、`invoke().to_messages()` 的区别？LCEL 用哪个？
3. `MessagesPlaceholder` 与 `.partial()` 各解决什么问题？举例。
4. `base_url` / `api_key` 的解析顺序？为什么 `load_dotenv()` 必须在 `ChatOpenAI(...)` 之前？
5. 写出 `prompt | llm | parser` 每环的数据类型，并说明 `StrOutputParser` 做了什么。
6. `RunnableParallel` 与 `RunnablePassthrough.assign` 的区别？RAG 为何更推荐后者？
7. `stream` 的 chunk 是什么类型？为什么不能直接当最终答案返回？
8. 只配 `OPENAI_API_BASE` 不配 `OPENAI_API_KEY`，`debug()` 会报什么错？在哪一行报？

<details>
<summary>参考答案要点</summary>

1. core = 抽象/协议；openai = 模型接入；community = 其他厂商。能跑，`debug()` 只用 core + `langchain_openai`；`flask-cors` 仍要单独装。
2. `format` 返回 `str` 只能用于文本模板；`to_string` 渲染带角色前缀的文本；`to_messages` 返回消息列表。LCEL 用 `invoke()`。
3. Placeholder 处理数量不定的历史消息；`partial` 预填固定变量（`now`、`format_instructions`）。
4. base：显式 → `OPENAI_API_BASE` → `OPENAI_BASE_URL` → 官方；key：显式 → `OPENAI_API_KEY`。构造时即校验凭证。
5. `dict → ChatPromptValue → AIMessage → str`；`StrOutputParser` 取 `.content`（str 透传，`{"text":...}` 取 text）。
6. Parallel 需重新组 dict 并显式取回原字段；assign 保留原输入全部字段只追加，扩展性更好。
7. `AIMessageChunk`（有 parser 时为 `str` 片段）；只是增量，需拼接或 SSE 逐块推。
8. `ChatOpenAI(model="deepseek-flash")` 这一行抛 Missing credentials。

</details>
