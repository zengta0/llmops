# 05 - LangChain 核心组件详解

> 知识来源：`第3周\第2章 实现联动第一个带UI的聊天机器人\llmops-api\study\` 下的 12 个脚本
> 落地代码：`internal/handler/app_handler.py` 的 `debug()`
> 前置阅读：`04-后端架构与请求链路.md`

## 读完你能做什么

- 说明 LangChain 解决什么问题，`langchain-core` / `langchain-openai` / `langchain-community` 各管什么；
- 熟练使用消息类型、Prompt 模板、ChatModel、OutputParser；
- 理解 `prompt | llm | parser` 的原理（`Runnable.__or__`）与每一环的数据类型；
- 用 `RunnableParallel` / `RunnablePassthrough.assign` 写出 RAG 最小雏形；
- 用回调观察链的内部过程，知道 LangSmith 怎么开；
- 逐行讲清项目里那 5 行核心代码，并避开高频坑。

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

## 二、消息体系

ChatModel 的输入输出都是消息列表：

| 类 | `type` | 角色 | 用途 |
| --- | --- | --- | --- |
| `SystemMessage` | `"system"` | 系统 | 人设、规则、当前时间 |
| `HumanMessage` | `"human"` | 用户 | 本轮提问 |
| `AIMessage` | `"ai"` | 助手 | 模型回复；多轮时要回填 |
| `ToolMessage` | `"tool"` | 工具 | 工具调用结果（本章不涉及） |

Prompt 模板里常用元组简写，LangChain 会自动转成对应消息类：`("system", "...")`、`("human", "{query}")`、`("ai", "...")`。

`AIMessage` 的结构（`study\2-Model组件及使用技巧\1.LLM与ChatModel使用技巧.py` 打印了 `type`、`content`、`response_metadata`）：

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `content` | `str` | 正文，最常取 |
| `type` | `str` | 固定 `"ai"` |
| `response_metadata` | `dict` | 厂商原始元数据：`model_name`、`finish_reason`、`token_usage`（各厂商不同） |
| `usage_metadata` | `dict \| None` | 统一 token 用量：`input_tokens`/`output_tokens`/`total_tokens`；新版 core 才有，部分兼容接口为 `None` |
| `id` / `tool_calls` | `str` / `list` | 消息 id、工具调用请求（本章不涉及） |

`AIMessage` 不适合直接返回前端：结构不 JSON 友好且带元数据。业务通常只要 `content`——这正是 OutputParser 的职责。

---

## 三、Prompt 组件

### 3.1 三种取结果方式

`study\1-Prompt组件及使用技巧\1.Prompt组件基础用法.py`：

```python
prompt = PromptTemplate.from_template("请讲一个关于{subject}的冷笑话")
prompt_value = prompt.invoke({"subject": "程序员"})

print(prompt.format(subject="喜剧演员"))
print(prompt_value.to_string())
print(prompt_value.to_messages())
```

| 写法 | 返回 | 说明 |
| --- | --- | --- |
| `prompt.format(...)` | `str` | 老 API，直接渲染字符串 |
| `prompt.invoke({...})` | `PromptValue` | 统一协议入口，**可参与 LCEL** |
| `.to_string()` | `str` | 渲染为文本（聊天模板带 `Human:` 等前缀） |
| `.to_messages()` | `list[BaseMessage]` | 转消息列表，可直接喂 ChatModel |

`format()` 与 `to_string()` 输出相似，但前者不能接链。**写 LCEL 一律用 `invoke()`。** `{subject}` 不是 f-string，真正替换发生在 `invoke`；键名必须与变量名完全一致，否则 `KeyError`。

### 3.2 ChatPromptTemplate + MessagesPlaceholder + partial

```python
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是OpenAI开发的聊天机器人，请根据用户的提问进行回复，当前的时间为:{now}"),
    MessagesPlaceholder("chat_history"),   # 历史消息数量不定，用占位符
    HumanMessagePromptTemplate.from_template("请讲一个关于{subject}的冷笑话"),
]).partial(now=datetime.now())

chat_prompt_value = chat_prompt.invoke({
    "chat_history": [("human", "我叫慕小课"), AIMessage("你好，我是ChatGPT，有什么可以帮到您")],
    "subject": "程序员",
})
print(chat_prompt_value.to_string())
```

- **`from_messages([...])`**：每项可以是元组、显式消息模板对象（如 `HumanMessagePromptTemplate.from_template`）、或 `MessagesPlaceholder`；
- **`MessagesPlaceholder("chat_history")`**：表示「这里插入数量不定的历史消息」，多轮对话必需；
- **`.partial(now=datetime.now())`**：预填变量并返回新模板（不可变），调用方不必重复传 `now`。

```text
System: ……当前的时间为:2024-06-08 21:36:00.123456
Human: 我叫慕小课
AI: 你好，我是ChatGPT，有什么可以帮到您
Human: 请讲一个关于程序员的冷笑话
```

`to_string()` 只用于调试或喂纯文本模型；给 `ChatOpenAI` 时传 `PromptValue` 本身即可保留消息结构。

### 3.3 拼接：字符串 `+` 与模板 `+`

`study\1-Prompt组件及使用技巧\2.字符串提示拼接.py`：

```python
prompt = (
        PromptTemplate.from_template("请讲一个关于{subject}的冷笑话")
        + ",让我开心下" +
        "\n使用{language}语言"
)
print(prompt.invoke({"subject": "程序员", "language": "中文"}).to_string())
```

`PromptTemplate + str` 走 `__add__`；`str + PromptTemplate` 走 `__radd__`。结果仍是**模板对象**，`input_variables` 是并集 `{"subject","language"}`。别当普通字符串拼接——先 `format` 再 `+` 会丢变量。

`study\1-Prompt组件及使用技巧\3.消息提示模板拼接.py`：

```python
chat_prompt = system_chat_prompt + human_chat_prompt
```

两个 ChatPromptTemplate 相加 = **消息列表拼接**（左前右后）。价值是把「人设」「历史」「用户输入」拆成独立片段按需组合。

### 3.4 复用：PipelinePromptTemplate

`study\1-Prompt组件及使用技巧\4.复用提示模板.py`：

```python
full_template = PromptTemplate.from_template("""{instruction}

{example}

{start}""")

instruction_prompt = PromptTemplate.from_template("你正在模拟{person}")
example_prompt = PromptTemplate.from_template("下面是一个交互例子：\n\nQ: {example_q}\nA: {example_a}")
start_prompt = PromptTemplate.from_template("现在，你是一个真实的人，请回答用户的问题:\n\nQ: {input}\nA:")

pipeline_prompt = PipelinePromptTemplate(final_prompt=full_template, pipeline_prompts=[
    ("instruction", instruction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
])
print(pipeline_prompt.invoke({...}).to_string())
```

机制：`final_prompt` 是骨架，三个槽位各由一个子模板渲染后填入。适合「多个业务共用角色设定与示例，只是结尾不同」。新版 `langchain-core` 仍可用但非主推，生产更常用 `ChatPromptTemplate + partial` 直接组合，理解思想即可。

---

## 四、Model 组件

### ChatOpenAI 与 LLM

`study` 里变量都叫 `llm`，但实例都是 `ChatOpenAI`：

| | LLM（文本补全） | ChatModel（聊天） |
| --- | --- | --- |
| 接口 | `/v1/completions` | `/v1/chat/completions` |
| 输入 | `str` | 消息列表 / `PromptValue` |
| 输出 | `str` | `AIMessage` |
| 类 | `langchain_openai.OpenAI`（旧称 `LLM`） | `langchain_openai.ChatOpenAI` |
| 现状 | 老模型 | **主流，本项目用这个** |

`llm.invoke(...)` 的入参可以是 `PromptValue`、消息列表或字符串（字符串自动包成 `HumanMessage`）。

### 三种调用方式

| 方法 | 返回 | 场景 |
| --- | --- | --- |
| `invoke(x)` | 单个结果 | 一次一问一答（`debug()` 用它） |
| `batch([x1, x2])` | 列表 | 多条独立请求，内部**并行**，吞吐高 |
| `stream(x)` | 迭代器（逐块） | 打字机效果、首字延迟敏感 |

**batch**（`study\2-Model组件及使用技巧\2.Model批处理.py`）：

```python
ai_messages = llm.batch([
    prompt.invoke({"query": "你好，你是?"}),
    prompt.invoke({"query": "请讲一个关于程序员的冷笑话"}),
])
for ai_message in ai_messages:
    print(ai_message.content)
```

默认线程池并发（`config={"max_concurrency": n}` 可限流），返回顺序与输入一致。

**stream**（`study\2-Model组件及使用技巧\3.Model流式输出.py`）：

```python
response = llm.stream(prompt.invoke({"query": "你能简单介绍下LLM和LLMOps吗?"}))
for chunk in response:
    print(chunk.content, flush=True, end="")
```

- 产出的是 **`AIMessageChunk`**，不是 `AIMessage`；`chunk.content` 是**增量片段**，拼起来才是完整回答；
- `flush=True` 让终端立即刷新；`chunk1 + chunk2` 可合并。

### 模型参数与简化

| 参数 | 含义 | 本项目 |
| --- | --- | --- |
| `model` | 模型名，原样进请求体 | `deepseek-flash` |
| `temperature` | 采样温度 | 未传 |
| `max_tokens` | 最大生成 token | 未传 |
| `timeout` / `max_retries` | 超时 / 重试 | 未传 |
| `base_url` / `api_key` | 接口地址 / 密钥 | 未传，读 env |

教学阶段只传 `model` 够用；生产至少显式设置 `timeout` 与 `max_retries`。

### 环境变量解析顺序（重点）

```text
base_url: 显式参数 → OPENAI_API_BASE → OPENAI_BASE_URL → https://api.openai.com/v1
api_key : 显式参数 → OPENAI_API_KEY → 都没有则报错
```

三个后果：**构造时校验凭证**（`ChatOpenAI(...)` 这一行就报 Missing credentials，不是等到 `invoke`）；**`.env` 必须先加载**（`dotenv.load_dotenv()` 在 `ChatOpenAI(...)` 之前）；**两个 base 变量别混**（本项目用 `OPENAI_API_BASE`，但有些库只认 `OPENAI_BASE_URL`，最稳妥是显式传参）：

```python
llm = ChatOpenAI(model="deepseek-flash",
                 base_url=os.getenv("OPENAI_API_BASE"),
                 api_key=os.getenv("OPENAI_API_KEY"))
```

> 本地 `.env` 只有 `OPENAI_API_BASE`，**没有 `OPENAI_API_KEY`**。跑 `debug()` 前补 `OPENAI_API_KEY=<你的密钥>`，否则构造即报错。

---

## 五、OutputParser

### StrOutputParser

`study\3-OutputParser组件及使用技巧\1.StrOutputParser使用技巧.py`：

```python
parser = StrOutputParser()
content = parser.invoke(llm.invoke(prompt.invoke({"query": "你好，你是?"})))
print(content)
```

职责：`AIMessage` → `.content`。也接受普通字符串（幂等）和 `{"text": "..."}`（取 `text`）。在链尾把 `AIMessage` 降维成 `str`，接口才能直接塞进 `{"content": content}`。

### JsonOutputParser

`study\3-OutputParser组件及使用技巧\2.JsonOutputParser使用技巧.py`：

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Joke(BaseModel):
    joke: str = Field(description="回答用户的冷笑话")
    punchline: str = Field(description="这个冷笑话的笑点")

parser = JsonOutputParser(pydantic_object=Joke)
prompt = ChatPromptTemplate.from_template(
    "请根据用户的提问进行回答。\n{format_instructions}\n{query}"
).partial(format_instructions=parser.get_format_instructions())

llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
joke = parser.invoke(llm.invoke(prompt.invoke({"query": "请讲一个关于程序员的冷笑话"})))
print(type(joke))          # <class 'dict'>
```

三个机制：`pydantic_object` 是**结构说明书**，返回值是 **`dict`** 而不是 Pydantic 对象；**`get_format_instructions()` 必须塞进提示词**，否则模型不知道要输出 JSON；`.partial(...)` 让说明只算一次，每次只传 `query`。整体链路：问题 → 渲染「格式说明 + 问题」 → 模型输出 JSON 文本 → 解析成 dict。

> `langchain_core.pydantic_v1` 是过渡模块，新版改为 `from pydantic import BaseModel, Field`。

### 解析失败的常见原因

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `Invalid json output` | 没注入 `format_instructions` | 用 `partial` 注入 |
| 外面包了 ```` ```json ```` | 模型加了 Markdown | `JsonOutputParser` 一般能容忍；自定义 parser 先剥壳 |
| 前后有解释文字 | 约束不足 | 提示词强调「只输出 JSON」 |
| JSON 被截断 | `max_tokens` 太小 | 调大或缩短输出 |
| 字段名不符 | 模型自由发挥 | `Field(description=...)` 写清语义 |
| 引号未转义 | 生成非法 JSON | 强调转义或换更稳的模型 |

---

## 六、LCEL 与 Runnable

### 6.1 手写 Chain：链的本质是循环

`study\4-LCEL表达式与Runnable可运行协议\1.手写Chain实现简易版本.py`：

```python
class Chain:
    def __init__(self, steps: list):
        self.steps = steps

    def invoke(self, input: Any) -> Any:
        for step in self.steps:
            input = step.invoke(input)   # 上一步输出 = 下一步输入
        return input

print(Chain([prompt, llm, parser]).invoke({"query": "你好，你是?"}))
```

秘密只有四条：链是**有序列表**；每步都是「有 `invoke` 方法的对象」；上一步输出 = 下一步输入；数据每步变形：`dict → PromptValue → AIMessage → str`。LangChain 把「有 `invoke`」正式命名为 **`Runnable` 协议**，实现了它就能进链。

### 6.2 `|` 是运算符重载

`study\4-LCEL表达式与Runnable可运行协议\2.LCEL表达式简化版本.py`：

```python
chain = prompt | llm | parser
print(chain.invoke({"query": "请讲一个程序员的冷笑话"}))
```

```python
a | b  ==  RunnableSequence(first=a, last=b)
```

`|` 调用 `Runnable.__or__`；`a | b | c` 左结合成 `RunnableSequence(a, b, c)`。序列自身也是 Runnable，可继续接。

### 6.3 数据流与链上三种调用

```text
{"query": "你好"}
   │ ChatPromptTemplate.invoke
   ▼
ChatPromptValue（[HumanMessage("你好")]）
   │ ChatOpenAI.invoke
   ▼
AIMessage（content + metadata）
   │ StrOutputParser.invoke
   ▼
"你好，我是……"
```

| 方法 | 语义 | 注意 |
| --- | --- | --- |
| `chain.invoke(x)` | 一份输入走完整条链 | 最常用 |
| `chain.batch([x1,x2])` | 多份输入并行 | 步骤需可并行 |
| `chain.stream(x)` | 流式 | 只有可流式环节逐块产出；parser 会逐块给 `str` |

### 6.4 RunnableParallel：一份输入多个出口

`study\5-两个Runnable核心类的讲解与使用\1.RunnableParallel使用技巧.py`：

```python
joke_chain = joke_prompt | llm | parser
poem_chain = poem_prompt | llm | parser

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)
res = map_chain.invoke({"subject": "程序员"})
print(res)   # {"joke": "...", "poem": "..."}
```

- 同一份输入**广播**给每个子链，子链**并行**执行，输出 dict（键就是构造时的键名）；
- `RunnableParallel(a=..., b=...)` 与 `RunnableParallel({"a": ..., "b": ...})` 等价。

### 6.5 模拟检索：RAG 最小雏形

`study\5-两个Runnable核心类的讲解与使用\2.RunnableParallel模拟检索.py`：

```python
from operator import itemgetter

def retrieval(query: str) -> str:
    """一个模拟的检索器函数"""
    print("正在检索:", query)
    return "我是慕小课"

prompt = ChatPromptTemplate.from_template(
    "请根据用户的问题回答，可以参考对应的上下文进行生成。\n\n<context>\n{context}\n</context>\n\n用户的提问是: {query}")

chain = {
    "context": lambda x: retrieval(x["query"]),
    "query": itemgetter("query"),
} | prompt | ChatOpenAI(model="gpt-3.5-turbo-16k") | StrOutputParser()

content = chain.invoke({"query": "你好，我是谁?"})
```

拆解：**dict 字面量**参与 `|` 时被自动转成 `RunnableParallel`（`coerce_to_runnable`），键是输出名；`lambda x: retrieval(x["query"])` 被包成 `RunnableLambda`，接收整个 dict、返回上下文；`itemgetter("query")` 同样被包成 Runnable，等价 `lambda x: x["query"]` 但更简洁。

```text
{"query": "你好，我是谁?"}
   ├─ context → retrieval(...) → "我是慕小课"
   └─ query   → itemgetter    → "你好，我是谁?"
{"context": "...", "query": "..."} → prompt → llm → parser
```

这就是 RAG 骨架：**先检索上下文，再把上下文和问题一起塞进提示词**。真实项目把 `retrieval` 换成向量库检索。

### 6.6 RunnablePassthrough.assign

`study\5-两个Runnable核心类的讲解与使用\3.RunnablePassthrough简化invoke调用.py`：

```python
from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough.assign(context=lambda x: retrieval(x["query"])) | prompt | llm | parser
content = chain.invoke({"query": "你好，我是谁?"})
```

| 版本 | 第一环 | 结果 |
| --- | --- | --- |
| 模拟检索 | `{"context": ..., "query": itemgetter("query")}` | 重新组 dict，必须显式取回 `query` |
| assign | `RunnablePassthrough.assign(context=...)` | **保留原输入所有键**，只追加 `context` |

`RunnablePassthrough` 原样输出输入，`.assign(**kwargs)` 在此基础上把计算结果写入输出 dict。输入字段再多也不用逐个取，**RAG 链最常用这种写法**。

普通函数（`lambda`、`def`）、`itemgetter` 都会被包成 `RunnableLambda`；约束是最好只接收一个位置参数（整个输入）。

---

## 七、回调与可观测性

### StdOutCallbackHandler

`study\6-利用回调功能调试链应用-让过程更透明\1.回调功能使用技巧.py`：

```python
resp = chain.stream(
    "你好，你是？",
    config={"callbacks": [StdOutCallbackHandler(), LLMOpsCallbackHandler()]}
)
for chunk in resp:
    pass
```

它打印链上每环的输入输出、提示词、模型原始响应，排查「模型到底收到了什么」很有效。传法两种，等价：

```python
chain.invoke(x, config={"callbacks": [handler]})          # 调用时
chain.with_config({"callbacks": [handler]}).invoke(x)     # 链上绑定
```

### 自定义 Handler 统计耗时

```python
class LLMOpsCallbackHandler(BaseCallbackHandler):
    """自定义LLMOps回调处理器"""
    start_at: float = 0

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> Any:
        print("messages:", messages)      # 最终发给模型的消息列表
        self.start_at = time.time()

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> Any:
        print("完整输出:", response)
        print("程序消耗:", time.time() - self.start_at)
```

- 两时间戳相减 = 模型耗时，可上报监控；`run_id`/`parent_run_id` 把回调串成调用树；
- 注意：`on_chat_model_start` 只对 ChatModel 触发；纯文本 LLM 走 `on_llm_start`。

### LangSmith

开启后每次链执行都会上传完整 trace。需要的变量：

```ini
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<你的密钥>
LANGCHAIN_PROJECT=llmops-api
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

现状：课程 `.env` 有 `LANGCHAIN_ENDPOINT`、`LANGCHAIN_API_KEY`、`LANGCHAIN_PROJECT`，但**没有** `LANGCHAIN_TRACING_V2`；本地 `.env` **一项都没有**。所以当前未开启追踪，想开四行都要补（缺 `TRACING_V2` 不会上传）。

---

## 八、对齐项目代码：逐行拆 `debug()`

```python
prompt = ChatPromptTemplate.from_template("{query}")
llm = ChatOpenAI(model="deepseek-flash")
parser = StrOutputParser()
chain = prompt | llm | parser
content = chain.invoke({"query": req.query.data})
return success_json({"content": content})
```

| 代码 | 对应 study | 说明 |
| --- | --- | --- |
| `ChatPromptTemplate.from_template("{query}")` | `3-OutputParser/1.StrOutputParser使用技巧.py` 第 16 行 | 只有 `HumanMessage` 占位的最简模板 |
| `ChatOpenAI(model="deepseek-flash")` | 各脚本的 `ChatOpenAI(model="gpt-3.5-turbo-16k")` | 只换 model；base_url/key 从 `.env` 读 |
| `StrOutputParser()` | 同上第 22 行 | `AIMessage` → `str` |
| `chain = prompt \| llm \| parser` | `4-LCEL/2.LCEL表达式简化版本.py` 第 22 行 | 写法完全一致 |
| `chain.invoke({"query": req.query.data})` | 同上第 25 行 | 校验后的用户输入 |

**唯一实质差别是模型名**（study 用 `gpt-3.5-turbo-16k`，项目用 `deepseek-flash`，其余只多了一层 `success_json` 包装）。DeepSeek 提供 OpenAI 兼容接口，`ChatOpenAI` 把 `base_url` 指过去、换成对方模型名即可，**上层链一行不改**——这就是抽象的价值。

补充两点：`debug(app_id)` 里的 `app_id` **未参与业务**，只用路由层 uuid 校验，后续章节才会 `app_service.get_app(app_id)`；`print('生成内容:', content)` 是临时调试，正式环境换结构化日志。

---

## 九、扩展阅读：换厂商只换模型类

`study\消息列表.py` 只有两行：

```python
from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
chat_model = QianfanChatEndpoint()
```

它想表达：只要都实现 `Runnable` 接口，`ChatOpenAI` / `QianfanChatEndpoint` / `ChatOllama` 之间可以互换，`chain = prompt | llm | parser` 之后完全一样。**换模型 = 换一行构造代码**，这也是项目要把「模型构建」与「链编排」分开写的原因。注意：`QianfanChatEndpoint` 需装 `langchain-community` 并配 `QIANFAN_AK`/`QIANFAN_SK`；各厂商对 system 消息、流式格式支持不完全一致，个别场景仍需适配。

---

## 十、组件速查表

| 环节 | 组件 | 输入 | 输出 | 脚本 |
| --- | --- | --- | --- | --- |
| 提示词 | `PromptTemplate`/`ChatPromptTemplate` | `dict` | `PromptValue` | `study/1-*` |
| 模型 | `ChatOpenAI` | `PromptValue`/消息列表 | `AIMessage`/`AIMessageChunk` | `study/2-*` |
| 解析 | `StrOutputParser`/`JsonOutputParser` | `AIMessage` | `str`/`dict` | `study/3-*` |
| 编排 | `RunnableSequence`（`\|`） | 任意 | 任意 | `study/4-*` |
| 并行/透传 | `RunnableParallel` / `RunnablePassthrough.assign` | `dict` | `dict` | `study/5-*` |
| 观测 | `BaseCallbackHandler` | 事件 | —— | `study/6-*` |

---

## 十一、常见坑

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

## 十二、自测题（8 道）

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
