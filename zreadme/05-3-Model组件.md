# 05-3 Model 组件

> 返回总览：[05-LangChain核心组件详解.md](./05-LangChain核心组件详解.md)｜上一节：[05-2-Prompt组件.md](./05-2-Prompt组件.md)｜下一节：[05-4-OutputParser.md](./05-4-OutputParser.md)
> 本节重点：`invoke` / `batch` / `stream` 三种调用，以及环境变量解析；附手写 openai SDK 的等价代码。

## 一、LangChain 的 Model 组件

### 1.1 ChatOpenAI 与 LLM

代码里变量常写作 `llm`，但实例都是 `ChatOpenAI`：

| | LLM（文本补全） | ChatModel（聊天） |
| --- | --- | --- |
| 接口 | `/v1/completions` | `/v1/chat/completions` |
| 输入 | `str` | 消息列表 / `PromptValue` |
| 输出 | `str` | `AIMessage` |
| 类 | `langchain_openai.OpenAI`（旧称 `LLM`） | `langchain_openai.ChatOpenAI` |
| 现状 | 老模型 | **主流，本项目用这个** |

`llm.invoke(...)` 的入参可以是 `PromptValue`、消息列表或字符串（字符串自动包成 `HumanMessage`）。

### 1.2 三种调用方式

| 方法 | 返回 | 场景 |
| --- | --- | --- |
| `invoke(x)` | 单个结果 | 一次一问一答（`debug()` 用它） |
| `batch([x1, x2])` | 列表 | 多条独立请求，内部**并行**，吞吐高 |
| `stream(x)` | 迭代器（逐块） | 打字机效果、首字延迟敏感 |

**batch**：

```python
ai_messages = llm.batch([
    prompt.invoke({"query": "你好，你是?"}),
    prompt.invoke({"query": "请讲一个关于程序员的冷笑话"}),
])
for ai_message in ai_messages:
    print(ai_message.content)
```

默认线程池并发（`config={"max_concurrency": n}` 可限流），返回顺序与输入一致。

**stream**：

```python
response = llm.stream(prompt.invoke({"query": "你能简单介绍下LLM和LLMOps吗?"}))
for chunk in response:
    print(chunk.content, flush=True, end="")
```

- 产出的是 **`AIMessageChunk`**，不是 `AIMessage`；`chunk.content` 是**增量片段**，拼起来才是完整回答；
- `flush=True` 让终端立即刷新；`chunk1 + chunk2` 可合并。

### 1.3 模型参数与简化

| 参数 | 含义 | 本项目 |
| --- | --- | --- |
| `model` | 模型名，原样进请求体 | `deepseek-flash` |
| `temperature` | 采样温度 | 未传 |
| `max_tokens` | 最大生成 token | 未传 |
| `timeout` / `max_retries` | 超时 / 重试 | 未传 |
| `base_url` / `api_key` | 接口地址 / 密钥 | 未传，读 env |

教学阶段只传 `model` 够用；生产至少显式设置 `timeout` 与 `max_retries`。

### 1.4 环境变量解析顺序（重点）

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

## 二、不用 LangChain 时：自己调 `client.chat.completions`

同一个模型，手写 openai SDK 的三种调用方式：

```python
import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)
```

**① invoke → `create`**

```python
resp = client.chat.completions.create(
    model="deepseek-flash",
    messages=[{"role": "user", "content": "你好，你是?"}],
    temperature=0.7,
)
content = resp.choices[0].message.content or ""
```

**② batch → 自己开线程池**

```python
def one(question: str) -> str:
    r = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": question}],
    )
    return r.choices[0].message.content or ""


with ThreadPoolExecutor(max_workers=2) as pool:
    answers = list(pool.map(one, ["你好，你是?", "请讲一个关于程序员的冷笑话"]))
```

`pool.map` 保证返回顺序与输入一致，`max_workers` 就是 LangChain 的 `max_concurrency`。

**③ stream → `stream=True` + 自己取 `delta.content`**

```python
for chunk in client.chat.completions.create(
    model="deepseek-flash",
    messages=[{"role": "user", "content": "你能简单介绍下LLM和LLMOps吗?"}],
    stream=True,
):
    delta = chunk.choices[0].delta.content
    if delta:                      # 首块/尾块的 content 可能是 None
        print(delta, end="", flush=True)
```

逐项对照：

| 能力 | LangChain | 手写 openai SDK |
| --- | --- | --- |
| 单次调用 | `llm.invoke(prompt_value)` 自动转消息 | `create(...)`，消息列表自己组 |
| 批量 | `llm.batch([...])`，一步并行 | 自己 `ThreadPoolExecutor` + `pool.map` |
| 限流 | `config={"max_concurrency": n}` | `max_workers=n` |
| 流式 | `llm.stream(x)` 产出 `AIMessageChunk`，chunk 可相加 | `stream=True`，自己拼 `delta.content`，自己判空 |
| 换模型 | 换模型类（`ChatOpenAI` → `ChatQianfan`），调用代码不变 | 换 SDK、换参数名、换结果解析路径 |
| 参数名 | 统一 `model` / `temperature` / `max_tokens` / `timeout` | 各厂商不完全一致，要查文档 |
| 凭证校验时机 | 构造 `ChatOpenAI(...)` 时即报错 | `OpenAI(...)` 构造时即报错（同为 openai SDK 行为） |

补充一点：`langchain_openai` 底层就是 openai SDK，所以 `OPENAI_API_BASE` / `OPENAI_API_KEY` 的解析顺序与手写版完全一致——这部分不是 LangChain 提供的便利，两边都要注意 `.env` 先加载。
