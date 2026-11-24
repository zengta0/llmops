# 05-8 记忆模块与 Runnable 进阶

> 返回总览：[05-LangChain核心组件详解.md](./05-LangChain核心组件详解.md)｜上一节：[05-7-落地与换厂商.md](./05-7-落地与换厂商.md)
> 参考源码：`第4周\第2章 LLMOps记忆模块解析与Runnable深入学习\llmops-api\study\7~19`
> 本地实现：`../internal/core/memory.py`、`../internal/handler/app_handler.py` 的 `debug()`（`app_handler_1.py` 保留单轮旧版作对照）
> 本节重点：模型为什么「没有记忆」、三种记忆策略、`RunnableWithMessageHistory`，以及 `bind` / `configurable_*` / `with_retry` / `with_fallbacks` / `with_listeners`。
> 本地状态：**已完成迁移**（LangChain 1.x 写法，未装新库、未降级）。第二~七节是课程 0.2.x 源码解读，其中大部分 API 在本环境已被**移除**，只作原理理解；本地实际实现见**第八节**，验收步骤见**第十二节**。

## 读完你能做什么

- 说清「模型本身无状态」是怎么回事，历史到底存在哪、怎么进 prompt；
- 不用 LangChain，手写一版「摘要缓冲记忆」；
- 区分 `ChatMessageHistory` / `BaseChatMemory` / 具体 Memory 三者职责；
- 按场景选 Buffer / TokenBuffer / SummaryBuffer / Entity 四种记忆；
- 用 `RunnableWithMessageHistory` 按 `session_id` 隔离多会话；
- 用 Runnable 进阶方法做「固参、热改配置、热换模型、重试、降级、生命周期钩子」；
- 逐行读懂课程版 `debug()` 的记忆链路，并知道它的局限。

---

## 零、先看这张表：本地版本与「已过时 / 已移除」API

> 实测环境（`llmops-api\venv`）：`langchain 1.5.0`、`langchain-core 1.8.4`、`langchain-openai 1.8.5`、`openai 3.58.0`，**未安装** `langchain-community`。

| 课程里出现的写法 | 本地状态 | 说明 |
| --- | --- | --- |
| `from langchain.memory import ConversationBufferWindowMemory` | ❌ 模块已被移除 | `langchain.memory` 在 1.x 不存在，导入即 `ModuleNotFoundError` |
| `ConversationTokenBufferMemory` / `ConversationSummaryBufferMemory` / `ConversationEntityMemory` | ❌ 同上 | 都住在 `langchain.memory` 里 |
| `from langchain.memory.chat_memory import BaseChatMemory` | ❌ 模块已被移除 | 手动 `load_memory_variables` / `save_context` 那一套随之消失 |
| `from langchain.chains import ConversationChain`、`langchain.chains.conversation.base` | ❌ 模块已被移除 | `langchain.chains` 在 1.x 不存在 |
| `from langchain_community.chat_message_histories import FileChatMessageHistory` | ❌ 依赖未安装 | `langchain-community` 不在本项目依赖里 |
| `from langchain_core.chat_history import BaseChatMessageHistory` | ✅ 可用 | 只有 `clear` 带 `@abstractmethod`，必须实现 |
| `from langchain_core.chat_history import InMemoryChatMessageHistory` | ✅ 可用 | 内存版历史仓库 |
| `from langchain_core.runnables import RunnableWithMessageHistory` | ⚠️ **可用但已过时** | 运行会打印 `LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.`，功能正常 |
| `RunnablePassthrough.assign` / `with_listeners` / `bind` / `configurable_fields` / `configurable_alternatives` / `with_retry` / `with_fallbacks` | ✅ 可用 | 第九节内容仍然成立 |
| `messages_from_dict` / `message_to_dict` | ✅ 可用 | JSON 与消息对象互转 |
| `langchain_core.messages.trim_messages` | ✅ 可用 | 按 token 裁剪消息的现代替代，可替换「按轮数」裁剪 |

一句话结论：**第二~七节讲「原理 + 课程写法」，第八节讲「本项目实际怎么写」**。想跑通代码只看第八节和第十二节。

---

## 一、为什么需要记忆模块

大模型接口是**无状态**的：每次 `create` 只处理这次请求里的 `messages`。你觉得它「记得」，是因为客户端每次把历史消息重新塞了进去。

```text
第 1 轮：messages = [user: 我叫慕小课]
第 2 轮：messages = [user: 我叫慕小课, assistant: 你好慕小课, user: 我叫什么？]
                    ↑ 上面这些历史，都是客户端自己存的
```

手写版最朴素的做法是一个 `list` 一直 `append`，但会碰到三个问题：

| 问题 | 表现 |
| --- | --- |
| 无限增长 | 聊得越久，`messages` 越长，token 费用线性上涨，最终超出上下文窗口 |
| 进程丢失 | 历史存在内存变量里，服务一重启就全没了 |
| 多用户串话 | 所有人共用一个 list，A 的历史会进 B 的 prompt |

记忆模块就是在解决这三件事：**存哪里**（内存/文件/数据库）、**留多少**（窗口/摘要）、**怎么变成 prompt 变量**（`history`）。

四种常见策略对比：

| 策略 | 保留什么 | 压缩方式 | 优点 | 代价 |
| --- | --- | --- | --- | --- |
| Buffer（全量缓冲） | 所有对话 | 不压缩 | 信息最全 | 必然超长 |
| Window（窗口） | 最近 k 轮 | 直接丢弃更早的 | 简单可控 | 早期信息彻底丢失 |
| SummaryBuffer（摘要缓冲） | 摘要 + 最近原文 | 旧对话让 LLM 总结 | 省 token 又保关键信息 | 每次摘要多花一次模型调用 |
| Entity（实体） | 按人/物建档 | LLM 抽取实体属性 | 适合长期用户画像 | 成本最高、实现最复杂 |

---

## 二、不用 LangChain 时：手写一版记忆（study 7）

课程 `7-Python+OpenAI原生SDK实现记忆功能\1.摘要缓冲混合记忆.py` 只依赖 `openai` SDK，把「摘要 + 缓冲」的逻辑全部手写：

```python
from typing import Any

from openai import OpenAI


class ConversationSummaryBufferMemory:
    """摘要缓冲混合记忆：摘要 + 最近若干轮原文"""

    def __init__(self, summary: str = "", chat_histories: list = None, max_tokens: int = 300):
        self.summary = summary                       # 1. 旧对话的摘要
        self.chat_histories = [] if chat_histories is None else chat_histories  # 2. 最近原文
        self.max_tokens = max_tokens                 # 3. 超过这个阈值就压缩
        self._client = OpenAI(base_url="https://api.xty.app/v1")

    @classmethod
    def get_num_tokens(cls, query: str) -> int:
        # 真实项目应换成 tiktoken；这里用字符数粗略估算
        return len(query)

    def save_context(self, human_query: str, ai_content: str) -> None:
        """每轮结束后调用：追加原文，超限就把最早一轮压进摘要"""
        self.chat_histories.append({"human": human_query, "ai": ai_content})

        if self.get_num_tokens(self.get_buffer_string()) > self.max_tokens:
            first_chat = self.chat_histories[0]
            self.summary = self.summary_text(
                self.summary,
                f"Human:{first_chat.get('human')}\nAI:{first_chat.get('ai')}",
            )
            del self.chat_histories[0]               # 压缩完才能丢弃

    def get_buffer_string(self) -> str:
        buffer = ""
        for chat in self.chat_histories:
            buffer += f"Human:{chat.get('human')}\nAI:{chat.get('ai')}\n\n"
        return buffer.strip()

    def load_memory_variables(self) -> dict[str, Any]:
        """把记忆变成一个可以塞进 prompt 的字符串"""
        return {
            "chat_history": f"摘要:{self.summary}\n\n历史信息:{self.get_buffer_string()}\n"
        }

    def summary_text(self, origin_summary: str, new_line: str) -> str:
        """让模型把「旧摘要 + 最早一轮对话」合并成新摘要"""
        prompt = f"""你是一个强大的聊天机器人，请根据用户提供的谈话内容总结摘要，
并将其添加到先前提供的摘要中，只返回新摘要。关键信息（姓名、爱好、性别、重要事件）必须保留。

当前摘要：{origin_summary}

新的对话：
{new_line}
"""
        completion = self._client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
```

配套的对话循环（同一文件下半部分）：

```python
client = OpenAI(base_url="https://api.xty.app/v1")
memory = ConversationSummaryBufferMemory("", [], 300)

while True:
    query = input("Human: ")
    if query == "q":
        break

    # 1. 取记忆 → 2. 手动拼进 prompt → 3. 请求模型 → 4. 存回记忆
    memory_variables = memory.load_memory_variables()
    answer_prompt = (
        "你是一个强大的聊天机器人，请根据对应的上下文和用户提问解决问题。\n\n"
        f"{memory_variables.get('chat_history')}\n\n"
        f"用户的提问是: {query}"
    )
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": answer_prompt}],
        stream=True,
    )

    ai_content = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is None:
            break
        ai_content += content
        print(content, flush=True, end="")
    memory.save_context(query, ai_content)
```

这版手写代码已经具备记忆模块的四个核心动作：

| 动作 | 手写版 | LangChain 对应 |
| --- | --- | --- |
| 存 | `chat_histories.append(...)` | `chat_memory.add_user_message/add_ai_message` |
| 取 | `load_memory_variables()` | `load_memory_variables({})` |
| 压 | `summary_text()` + `del chat_histories[0]` | `ConversationSummaryBufferMemory` 内部完成 |
| 拼 | f-string 塞进 `answer_prompt` | `MessagesPlaceholder("history")` |

手写版缺的三件事：**持久化**（重启即失）、**消息类型**（只有裸字符串，没有 role 结构）、**并发安全**（全局单例，多用户串话）。下面三节依次补上。

---

## 三、ChatMessageHistory：只负责「存」

`ChatMessageHistory` 把「存消息」单独抽出来，只管增删查，不做裁剪、不做摘要。

### 3.1 内存版（study 8-1）

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

chat_history = InMemoryChatMessageHistory()
chat_history.add_user_message("你好，我是慕小课，你是谁？")
chat_history.add_ai_message("你好，我是ChatGPT，有什么可以帮到您的？")

print(chat_history.messages)   # [HumanMessage(...), AIMessage(...)]
```

### 3.2 文件版（study 8-2）

```python
from langchain_community.chat_message_histories import FileChatMessageHistory

chat_history = FileChatMessageHistory("./memory.txt")
chat_history.add_user_message(query)
chat_history.add_ai_message(ai_content)
```

文件里存的是**整份 JSON 列表**，每条形如：

```json
{"type": "human", "data": {"content": "你好，我是慕小课", "additional_kwargs": {}, "type": "human"}}
```

所以它本质是「每次读取全量、追加后整体覆盖」。优点是不用数据库就能持久化；缺点是文件越大读写越慢，多进程并发写会互相覆盖。

> ⚠️ **本地不可用**：`langchain_community` 未安装，上面三行 import 在本项目会直接 `ModuleNotFoundError`。本地用第八节自己写的 `FileChatMessageHistory` 顶替，行为一致（整份 JSON 读写覆盖）。

---

## 四、BaseChatMemory：把「历史」变成「prompt 变量」

> ⚠️ **本地不可用（已移除）**：`langchain.memory.chat_memory` 在 1.x 不存在，本节只作原理理解。它负责的「load / save」在本地由 `RunnableWithMessageHistory` 自动完成（第八节）。

`ChatMessageHistory` 只是仓库，`BaseChatMemory` 负责把它翻译成链能用的键值（study 9）：

```python
from langchain.memory.chat_memory import BaseChatMemory

memory = BaseChatMemory(
    input_key="query",      # 从链输入 dict 里取哪个键当「用户说了什么」
    output_key="output",    # 从链输出里取哪个键当「AI 回了什么」
    return_messages=True,   # True → 返回消息对象列表；False → 返回拼好的字符串
)

history = memory.load_memory_variables({})   # {"history": [HumanMessage(...), ...]}
memory.save_context({"query": "我叫慕小课"}, {"output": "你好慕小课"})
```

两个关键方法：

| 方法 | 时机 | 干什么 |
| --- | --- | --- |
| `load_memory_variables(inputs)` | 调链**之前** | 读出历史，返回 `{"history": ...}` 供 `MessagesPlaceholder("history")` 使用 |
| `save_context(inputs, outputs)` | 调链**之后** | 把这一轮的输入/输出写进 `ChatMessageHistory` |

注意：`BaseChatMemory` **不会自动执行**。课程里靠两种方式触发：`Chain` 内置的 `memory` 参数，或者手动 `load` / `save`（study 10、11），或者用 `with_listeners(on_end=...)`（课程版 `debug()`）。

三个参数最容易配错：

- `input_key` / `output_key` 必须与 `invoke` / 输出字典的键**完全一致**，否则 `save_context` 取不到值直接 `KeyError`；
- `return_messages=True` 时返回 `list[BaseMessage]`，正好喂给 `MessagesPlaceholder`；若为 `False` 返回字符串，塞进 `MessagesPlaceholder` 会类型报错；
- `load_memory_variables({})` 返回的键名（默认 `history`）必须与模板里 `MessagesPlaceholder("history")` 的名字一致。

---

## 五、四种记忆策略（study 10~12）

> ⚠️ **本地不可用（已移除）**：本节四个类全部来自 `langchain.memory`，在本环境导入即报错。价值在于理解「窗口 / token / 摘要 / 实体」四种策略的取舍；本地只用「窗口」策略，而且是自己实现的（`window_rounds=3`，第八节）。

### 5.1 窗口记忆：只留最近 k 轮

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=3,                    # 只保留最近 3 轮
    input_key="query",
    output_key="output",
    return_messages=True,
    chat_memory=FileChatMessageHistory("./storage/memory/chat_history.txt"),  # 可选：落盘
)
```

适合客服问答这类「只关心最近上下文」的场景。`k` 按**轮**算（一问一答算一次 `save_context`）。

### 5.2 Token 窗口：按 token 数裁剪（study 10）

```python
from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI

memory = ConversationTokenBufferMemory(
    return_messages=True,
    input_key="query",
    llm=ChatOpenAI(),       # 用它来估算 token
)
```

比按轮数更精确，但每次 `load` 都要算 token，且依赖一个额外的 `llm` 对象。

### 5.3 摘要缓冲：摘要 + 最近原文（study 11）

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    max_token_limit=300,
    return_messages=True,
    input_key="query",
    llm=ChatOpenAI(model="gpt-3.5-turbo-16k"),   # 摘要由它生成
)
```

原理就是第二节手写版的官方实现：超过 `max_token_limit` 时，自动把最早的几轮交给 `llm` 总结成摘要，摘要替换原文，从而「压缩但不忘事」。

### 5.4 实体记忆：按人/物建档（study 12）

```python
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint

llm = QianfanChatEndpoint()     # 课程用千帆演示，也可以换 ChatOpenAI
chain = ConversationChain(
    llm=llm,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm),
)

chain.invoke({"input": "你好，我是慕小课。我最近正在学习LangChain。"})
chain.invoke({"input": "我最喜欢的编程语言是 Python。"})
print(chain.memory.entity_store.store)   # 按实体名归档的字典
```

它把对话里出现的**名词性实体**抽出来单独建档（谁、喜欢什么、住在哪），适合长期用户画像；代价是每轮都要额外调 LLM 做抽取。

选择建议：

```text
只要最近上下文        → Window（k 小一点）
要控制 token 且预算足 → TokenBuffer
要既省 token 又不丢关键信息 → SummaryBuffer
要做长期用户画像      → Entity（或自己接数据库）
```

---

## 六、RunnableWithMessageHistory：工程化封装（study 14）

> ⚠️ **已过时但仍可用**：本环境运行会打印 `LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.`。官方推荐迁移到 LangGraph 持久化（后续章节），但当前功能完全正常，本项目仍采用它。
> 另外注意：本节示例里的 `FileChatMessageHistory` 来自 `langchain_community`（本地未安装），本项目替换为第八节的自写版本。

前面几种 Memory 都有个共同麻烦：**手动 load / save**，而且不好按用户隔离。`RunnableWithMessageHistory` 把这两步收进包装器：

```python
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """按 session_id 取（或创建）该会话的历史仓库"""
    if session_id not in store:
        store[session_id] = FileChatMessageHistory(f"./chat_history_{session_id}.txt")
    return store[session_id]


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个强大的聊天机器人，请根据用户的需求回复问题。"),
    MessagesPlaceholder("history"),
    ("human", "{query}"),
])
chain = prompt | ChatOpenAI(model="gpt-3.5-turbo-16k") | StrOutputParser()

with_message_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,          # 关键：怎么按会话取历史
    input_messages_key="query",   # 用户输入在 dict 里的键
    history_messages_key="history",  # prompt 里占位符的名字
)

with_message_chain.stream(
    {"query": "我叫什么？"},
    config={"configurable": {"session_id": "muxiaoke"}},   # 会话隔离靠它
)
```

对比三种做法：

| 做法 | 谁负责 load/save | 能否多会话隔离 |
| --- | --- | --- |
| 手动 `memory.load/save`（study 10/11） | 业务代码 | 要自己写字典 |
| `with_listeners(on_end=save)`（课程版 debug） | 监听器 + 一个全局 memory | 不能（单文件全局） |
| `RunnableWithMessageHistory` + `session_id` | 包装器 | 能，`store` 按 id 分仓 |

`session_id` 通常来自登录用户 ID 或前端生成的会话 ID，**不要**让前端随便传一个别人的 id 就能读到对方历史。

---

## 七、逐行读懂课程版 `debug()`

> ⚠️ 这一版代码在本环境**跑不起来**：依赖的 `langchain.memory`、`BaseMemory`、`langchain_community` 全部缺失。读它是为了理解「历史怎么进 prompt、什么时候保存」，本地等价实现见第八节。

课程 `第4周\第2章\...\internal\handler\app_handler.py`：

```python
@classmethod
def _load_memory_variables(cls, input: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """从 config 里取出 memory，加载历史"""
    configurable = config.get("configurable", {})
    configurable_memory = configurable.get("memory", None)
    if configurable_memory is not None and isinstance(configurable_memory, BaseMemory):
        return configurable_memory.load_memory_variables(input)
    return {"history": []}


@classmethod
def _save_context(cls, run_obj: Run, config: RunnableConfig) -> None:
    """链跑完后，把输入/输出写回同一份 memory"""
    configurable = config.get("configurable", {})
    configurable_memory = configurable.get("memory", None)
    if configurable_memory is not None and isinstance(configurable_memory, BaseMemory):
        configurable_memory.save_context(run_obj.inputs, run_obj.outputs)


def debug(self, app_id: UUID):
    req = CompletionReq()
    if not req.validate():
        return validate_error_json(req.errors)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个强大的聊天机器人，能根据用户的提问回复对应的问题"),
        MessagesPlaceholder("history"),
        ("human", "{query}"),
    ])
    memory = ConversationBufferWindowMemory(
        k=3,
        input_key="query",
        output_key="output",
        return_messages=True,
        chat_memory=FileChatMessageHistory("./storage/memory/chat_history.txt"),
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k")

    chain = (RunnablePassthrough.assign(
        history=RunnableLambda(self._load_memory_variables) | itemgetter("history")
    ) | prompt | llm | StrOutputParser()).with_listeners(on_end=self._save_context)

    chain_input = {"query": req.query.data}
    content = chain.invoke(chain_input, config={"configurable": {"memory": memory}})

    return success_json({"content": content})
```

逐行对照：

| 代码 | 作用 | 坑 |
| --- | --- | --- |
| `MessagesPlaceholder("history")` | 留出历史消息的位置 | 名字必须和 `load` 返回的键一致 |
| `ConversationBufferWindowMemory(k=3, ...)` | 只留最近 3 轮，并落盘 | `output_key="output"` 必须和链输出对应 |
| `RunnablePassthrough.assign(history=...)` | 在执行 prompt 前，把 `history` 注入输入 dict | 不用 assign 就得手动拼 `{"query":..., "history":...}` |
| `RunnableLambda(self._load_memory_variables) \| itemgetter("history")` | 从 config 取 memory → load → 只取 `history` 字段 | 返回的是 `{"history": [...]}`，不取字段会给 prompt 多塞一个键 |
| `with_listeners(on_end=self._save_context)` | 链**成功结束**后保存上下文 | 异常结束走 `on_error`，不会保存 |
| `config={"configurable": {"memory": memory}}` | 把 memory 作为运行时配置传入 | 不传则 `_load` 返回空历史，退化成单轮 |

**为什么 memory 要走 `config.configurable` 而不是闭包？** 闭包也能跑，但 memory 变成硬编码依赖，无法替换、无法测试、无法按请求传不同用户的 memory。走 config 后，调用方决定用哪份记忆，链本身保持无状态。

这版实现有三个明显局限（也是后续章节要升级的点）：

1. `FileChatMessageHistory("./storage/memory/chat_history.txt")` 是**固定路径**，所有 `app_id`、所有用户共用一份历史；
2. `memory` 每次请求新建，但历史文件是共享的，高并发时读写会互相覆盖；
3. 没有用 `session_id`，无法做多会话隔离。

---

## 八、本地实现：LangChain 1.x 版记忆链路（已完成迁移）

> 落地代码：`../internal/core/memory.py`（新增）+ `../internal/handler/app_handler.py` 的 `debug()`
> 依赖：`langchain 1.5.0` / `langchain-core 1.8.4` / `langchain-openai 1.8.5`，**没有** `langchain-community`
> 原则：不装新库、不降级 langchain，只用 1.x 仍在的 `BaseChatMessageHistory` + `RunnableWithMessageHistory` 复刻课程效果。

### 8.1 整体设计：职责拆成三层

课程把「存储、编排、裁剪」全塞进 `ConversationBufferWindowMemory` 一个类；1.x 里这个类没了，我们就按职责自己拼：

```text
┌──────────────┐   ① 读历史填充 history    ┌───────────────────────────┐
│   debug()    │ ────────────────────────▶ │ RunnableWithMessageHistory │
│ (prompt|llm) │ ◀──────────────────────── │  （负责 load / save 时机）  │
└──────────────┘   ② 写回本轮问答           └─────────────┬─────────────┘
                                                          │ session_id
                                                          ▼
                                            ┌───────────────────────────┐
                                            │ FileChatMessageHistory     │
                                            │ JSON 落盘 + 3 轮窗口裁剪    │
                                            └───────────────────────────┘
```

| 层 | 谁负责 | 对应课程的什么 |
| --- | --- | --- |
| 存储（存哪里、留多少） | `FileChatMessageHistory` | `FileChatMessageHistory` + `ConversationBufferWindowMemory(k=3)` |
| 编排（何时读、何时写） | `RunnableWithMessageHistory` | `_load_memory_variables` + `_save_context` + `with_listeners(on_end=...)` |
| 使用（模板、会话 id） | `debug()` | 课程版 `debug()` |

### 8.2 `internal/core/memory.py` 逐块讲解

#### ① 导入与路径常量

```python
import json
import os
from pathlib import Path
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

MEMORY_STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage" / "memory"
```

- `parents[0]=core`、`parents[1]=internal`、`parents[2]=llmops-api`，所以落点永远是 `llmops-api\storage\memory`
- 必须写 `parents`（复数，可下标），写成 `.parent[2]` 会报 `TypeError: 'WindowsPath' object is not subscriptable`
- 不用课程那种 `"./storage/memory/..."`，因为相对路径会跟着启动目录变（`python -m app.http.app` 与 PyCharm Script 两种方式的 cwd 不同）

#### ② 构造：一个会话一个文件

```python
class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, storage_dir, session_id, window_rounds=3) -> None:
        self.storage_dir = Path(storage_dir)
        self.session_id = session_id
        self.window_rounds = window_rounds
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_dir / f"{self.session_id}.json"
```

- `mkdir(parents=True, exist_ok=True)`：目录不存在就建，已存在不报错，所以不用手动建 `storage/memory`
- 课程是**所有会话共用一个 `chat_history.txt`**；本地是 `<session_id>.json`，而 `session_id` 传的就是 `app_id` —— 多应用隔离靠这一行实现

#### ③ 读历史：`messages` 属性

```python
    @property
    def messages(self) -> list[BaseMessage]:
        if not self.file_path.exists():
            return []

        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        return messages_from_dict(raw)
```

- 必须是 `@property` 且名字是**复数** `messages`：`RunnableWithMessageHistory` 内部就是按 `history.messages` 读的，写成 `def message(self)` 会得到 `AttributeError: 'FileChatMessageHistory' object has no attribute 'messages'`
- 文件不存在（第一次聊天）返回 `[]`
- 文件损坏 / 读失败也返回 `[]`：记忆是附加功能，不该让聊天接口变成 500，下一轮写入会覆盖成合法内容

#### ④ 写历史 + 窗口裁剪：`add_messages`

```python
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages) + list(messages)

        if self.window_rounds > 0:
            max_messages = self.window_rounds * 2
            all_messages = all_messages[-max_messages:]

        self.file_path.write_text(
            json.dumps([message_to_dict(m) for m in all_messages], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
```

- 方法名也必须是**复数** `add_messages`；写成单数 `add_message` 时，包装器调用会 `AttributeError`
- 一轮 = 1 条 human + 1 条 ai = 2 条，所以 `k=3` 对应保留 `3 * 2 = 6` 条，`all_messages[-6:]` 取最后 6 条
- 先读旧、再拼接、再整体覆盖写，等价于课程的 `save_context` + `ConversationBufferWindowMemory` 的裁剪
- `ensure_ascii=False` 让中文在文件里直接可读；`indent=2` 方便你打开对照

#### ⑤ `clear` 与工厂函数

```python
    def clear(self) -> None:
        if self.file_path.exists():
            self.file_path.unlink()


def get_file_chat_message_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(MEMORY_STORAGE_DIR, session_id)
```

- `clear` 是 `BaseChatMessageHistory` 里**唯一带 `@abstractmethod` 的方法**，不实现连实例化都过不了；`messages` / `add_messages` 属于「约定要实现」，不写不会在实例化时报错，但运行时必然出问题
- 工厂函数是包装器要求的：每次调用它会用 `config["configurable"]["session_id"]` 来调这个函数拿历史对象

#### ⑥ 与课程版 `FileChatMessageHistory` 的差异

| 维度 | 课程（community 版） | 本地实现 |
| --- | --- | --- |
| 依赖 | `langchain-community` | 只需 `langchain-core` |
| 文件粒度 | 单文件全局，所有用户/app 共用 | 每个 `session_id` 一个文件 |
| 窗口裁剪 | 由 Memory 层做 | 存储层写入时做（`window_rounds`） |
| 坏文件 | 直接抛异常 | 返回空历史，自动恢复 |
| 编码 | 平台默认 | 显式 UTF-8，中文可读 |
| 初始化 | 传文件路径 | 传目录 + session_id |

### 8.3 `debug()` 的三处改动

**改动 1：import**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from internal.core.memory import get_file_chat_message_history
```

**改动 2：提示词从单串变三段**

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个强大的聊天机器人，能根据用户的提问回复对应的问题"),
    MessagesPlaceholder("history"),
    ("human", "{query}")
])
```

**改动 3：在链外包一层包装器**

```python
chain = prompt | llm | parser

conversation = RunnableWithMessageHistory(
    chain,
    get_file_chat_message_history,
    input_messages_key="query",
    history_messages_key="history"
)

content = conversation.invoke(
    {"query": req.query.data},
    config={"configurable": {"session_id": str(app_id)}},
)
```

参数对照（最容易配错的就是这三处）：

| 参数 | 值 | 必须与谁对应 |
| --- | --- | --- |
| `input_messages_key` | `"query"` | `invoke({"query": ...})` 的键 |
| `history_messages_key` | `"history"` | `MessagesPlaceholder("history")` 的名字 |
| `configurable.session_id` | `str(app_id)` | 记忆文件名 `<app_id>.json` |

`chain` 本身仍然是「无状态」的 `prompt | llm | parser`——记忆不是链的能力，是包装器在链外面加的行为。这一点和课程版把 `memory` 塞进 `config` 是同一个思路，只是本地换成了 `session_id`。

### 8.4 运行时行为（已实测）

- 第 2 轮送给模型的消息：`[system, human(第1轮提问), ai(第1轮回答), human(本轮提问)]`
- 连发 4 轮后文件里只剩 6 条（最近 3 轮），最早一轮被丢弃
- 换一个 `app_id` 请求，读到的是空历史 → 多应用隔离生效
- 控制台会出现一条 `LangChainDeprecationWarning`（`RunnableWithMessageHistory` 已过时）——警告不是错误，功能正常
- 每个文件内容形如：

```json
[
  {"type": "human", "data": {"content": "我叫小明", "additional_kwargs": {}, "response_metadata": {}, "type": "human"}},
  {"type": "ai", "data": {"content": "你好小明", "additional_kwargs": {}, "response_metadata": {}, "type": "ai"}}
]
```

### 8.5 与课程版 `debug()` 的逐点对照

| 能力 | 课程版 | 本地版 |
| --- | --- | --- |
| 调用前加载历史 | `RunnablePassthrough.assign(history=RunnableLambda(_load_memory_variables) \| itemgetter("history"))` | 包装器内部自动读取 `messages` |
| 调用后保存历史 | `with_listeners(on_end=_save_context)` | 包装器内部自动 `add_messages` |
| 历史传入方式 | `config["configurable"]["memory"]` 传 memory 对象 | `config["configurable"]["session_id"]` 传 id |
| 会话隔离 | 无，固定单文件 | 有，按 `app_id` 分文件 |
| 窗口裁剪 | `ConversationBufferWindowMemory(k=3)` | `window_rounds=3` 写入时切片 |
| 需要自己写的类 | 只有 `_load` / `_save` 两个静态方法 | `BaseChatMessageHistory` 子类 + 工厂函数 |
| 异常时是否保存 | `on_end` 不触发，不保存 | 链抛异常时同样不写入 |

### 8.6 已知限制与升级方向

1. **`RunnableWithMessageHistory` 已过时** —— 官方推荐 LangGraph 的 checkpointer/persistence，后续章节会替换；现在保留是因为改动最小、行为最接近课程。
2. **按轮裁剪而非 token** —— 想要更精确可用 `langchain_core.messages.trim_messages`，按 `max_tokens` 截断。
3. **文件存储无锁** —— 同一会话高并发时可能互相覆盖；生产应换 Redis / 数据库。
4. **`app_id` 未鉴权** —— 知道 UUID 就能读到该应用的历史，后续接入用户体系后要校验归属。
5. **历史里不含 token 用量** —— `AIMessage` 的 `usage_metadata` 能记录消耗，做计费时需要保留。

---

## 九、Runnable 进阶六件套（study 15~19）

这些方法都返回新的 Runnable，可以继续 `|`。

### 9.1 `bind`：把参数固化在链里（study 15）

```python
chain = prompt | ChatOpenAI(model="gpt-3.5-turbo").bind(model="gpt-4o") | StrOutputParser()
```

`bind` 的参数会**覆盖**组件同名参数。也可以给普通函数补默认参数：

```python
from langchain_core.runnables import RunnableLambda

def get_weather(location: str, unit: str, name: str) -> str:
    return f"{location}天气为{random.randint(24, 40)}{unit}"

runnable = RunnableLambda(get_weather).bind(unit="摄氏度", name="慕小课")
runnable.invoke("广州")    # 只传 location，其余由 bind 补齐
```

手写对照：`functools.partial(get_weather, unit="摄氏度")`，本质一样，只是 `bind` 返回的是 Runnable。

### 9.2 `configurable_fields`：运行时改链内部字段（study 16）

```python
from langchain_core.runnables import ConfigurableField

llm = ChatOpenAI(model="gpt-3.5-turbo-16k").configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="大模型的温度",
        description="越低越确定，越高越随机",
    )
)

chain = prompt | llm | StrOutputParser()
chain.invoke({"x": 1000}, config={"configurable": {"llm_temperature": 0}})
```

提示词也能这样换（study 16-2）：

```python
prompt = PromptTemplate.from_template("请写一篇关于{subject}主题的冷笑话").configurable_fields(
    template_demo=ConfigurableField(id="prompt_template"),
)
prompt.invoke(
    {"subject": "程序员"},
    config={"configurable": {"prompt_template": "请写一篇关于{subject}主题的藏头诗"}},
)
```

区别：`bind` 是**固化**（调用方改不了），`configurable_fields` 是**留口子**（调用方可覆盖）。`id` 拼错不会报错，配置会被静默忽略——这是最常见的坑。

### 9.3 `configurable_alternatives`：运行时换模型（study 17）

```python
llm = ChatOpenAI(model="gpt-3.5-turbo-16k").configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="gpt35",
    gpt4=ChatOpenAI(model="gpt-4o"),
    wenxin=QianfanChatEndpoint(),
)

chain.invoke(
    {"query": "你好，你是什么模型？"},
    config={"configurable": {"llm": "gpt4"}},   # 同一个键必须是上面定义过的 key
)
```

适合「同一业务按用户等级/租户切模型」，也方便做 A/B。

### 9.4 `with_retry`：失败重试（study 18-1）

```python
from langchain_core.runnables import RunnableLambda

chain = RunnableLambda(func).with_retry(stop_after_attempt=2)
```

注意 `stop_after_attempt` 是**总尝试次数**（含第一次），不是「重试几次」。手写对照：

```python
def call_with_retry(fn, times=2):
    for i in range(times):
        try:
            return fn()
        except Exception:
            if i == times - 1:
                raise
```

### 9.5 `with_fallbacks`：主模型挂了自动降级（study 18-2）

```python
llm = ChatOpenAI(model="gpt-3.5-turbo-18k").with_fallbacks([QianfanChatEndpoint()])
chain = prompt | llm | StrOutputParser()
chain.invoke({"query": "你好，你是?"})   # 主模型报错 → 自动用备用模型
```

手写对照是 `try: main() except: backup()`，`with_fallbacks` 的价值在于**链不用改**，降级关系挂在模型对象上。

### 9.6 `with_listeners`：生命周期钩子（study 19）

```python
def on_start(run_obj, config): ...
def on_end(run_obj, config): ...
def on_error(run_obj, config): ...

chain = RunnableLambda(lambda x: time.sleep(x)).with_listeners(
    on_start=on_start,
    on_end=on_end,
    on_error=on_error,
)

chain.invoke(2, config={"configurable": {"name": "慕小课"}})
```

| 钩子 | 时机 | 典型用途 |
| --- | --- | --- |
| `on_start` | 链开始执行 | 打点、记录请求 ID |
| `on_end` | 链**成功**结束 | 保存记忆、写审计日志 |
| `on_error` | 链抛异常 | 告警、补偿 |

`run_obj.inputs` / `run_obj.outputs` 就是这一轮的输入输出，课程版 `_save_context` 正是靠它拿到 `{"query": ...}` 和生成内容。注意 `on_end` **不会**在异常时触发，所以「保存记忆」和「异常补偿」要分开写在 `on_end` / `on_error` 里。

---

## 十、常见坑

**坑 1：占位符名字对不上**。`load_memory_variables` 返回 `{"history": ...}`，模板写 `MessagesPlaceholder("chat_history")`，运行时 `KeyError: chat_history`。

**坑 2：`return_messages=False` 却用 MessagesPlaceholder**。字符串不能当消息列表，报类型错误。想用占位符就必须 `return_messages=True`；反过来，模板里用 `{history}` 文本插值时才需要字符串。

**坑 3：`input_key` / `output_key` 与链的键不一致**。`memory.save_context` 读不到 `query` / `output`，直接 `KeyError`。课程版里 `output_key="output"` 能对上，是因为 `StrOutputParser` 的输出被包装成 `{"output": ...}`——如果用自定义输出结构，要跟着改。

**坑 4：把 `on_end` 当 `finally` 用**。异常时 `on_end` 不执行，记忆不会保存，看起来像「这轮白聊了」。

**坑 5：`FileChatMessageHistory` 固定路径 + 多用户**。课程版 `./storage/memory/chat_history.txt` 是全局单文件，两个用户同时聊会串话。工程上至少按 `user_id` 分文件，或用 `RunnableWithMessageHistory` + `session_id`。

**坑 6：`SummaryBuffer` 的隐形成本**。摘要由额外的 LLM 调用生成，`max_token_limit` 设得太小会频繁摘要，费用反而上升。

**坑 7：`configurable_fields` 的 `id` 拼错**。不报错、不生效，表现为「配置传了但结果没变」。排查：`llm.config_specs` 看已注册的 id。

**坑 8：`with_retry` 重试非幂等操作**。写数据库、扣费之类的步骤不要整体包 `with_retry`，否则可能重复执行。

**坑 9（本地版特有）：`messages` / `add_messages` 写成了单数**。`BaseChatMessageHistory` 只把 `clear` 标记为抽象方法，所以少写或写错名字**不会在实例化时报错**，而是在 `RunnableWithMessageHistory` 调用时抛 `AttributeError: 'FileChatMessageHistory' object has no attribute 'messages'`。另外 `messages` 必须加 `@property`，否则拿到的是函数对象。

**坑 10（本地版特有）：路径推导写成 `Path(__file__).parent[2]`**。`.parent` 返回单个 `Path`，不可下标，报 `TypeError: 'WindowsPath' object is not subscriptable`；要用 `.parents[2]`。

**坑 11（本地版特有）：按课程升级依赖**。执行 `pip install "langchain~=0.2.1"` 会把现有 1.x 降级，`RunnableWithMessageHistory`、`ChatOpenAI` 等行为与导入路径都可能变化。本项目固定用 1.x，遇到缺失的记忆类一律按第八节自写，不要装 `langchain-community` 来凑。

---

## 十一、自测题（8 道）

1. 模型「记得」上一轮对话，是模型有状态，还是别的原因？历史消息以什么形式进入请求？
2. 手写摘要缓冲记忆里，为什么必须「先摘要成功、再删最早一轮」？顺序反了会怎样？
3. `ChatMessageHistory`、`BaseChatMemory`、`ConversationBufferWindowMemory` 三者是什么关系？
4. `input_key`、`output_key`、`return_messages` 分别影响哪一步？各举一个配错后的报错。
5. Window、TokenBuffer、SummaryBuffer、Entity 四种记忆分别适合什么场景？各说一个缺点。
6. 课程版 `debug()` 为什么用 `RunnablePassthrough.assign` 注入 `history`，而不是直接 `{"query":..., "history":...}`？
7. `bind` 和 `configurable_fields` 都是「传参数」，区别是什么？什么时候必须用后者？
8. `with_listeners(on_end=...)` 保存记忆，在模型报错那一轮会发生什么？应该怎么补？

<details>
<summary>参考答案要点</summary>

1. 模型无状态；客户端每轮把历史消息（`list[dict]` 或消息对象）重新放进 `messages` 发给模型。
2. 先摘要再删除，保证最早一轮的信息已被压缩进 `summary`；顺序反了会直接丢信息。
3. `ChatMessageHistory` 只存消息；`BaseChatMemory` 在它之上提供 `load/save` 与键映射；具体 Memory 再叠加裁剪/摘要策略。
4. `input_key/output_key` 决定 `save_context` 从哪取输入输出（错则 `KeyError`）；`return_messages` 决定 load 返回消息对象还是字符串（错则占位符类型报错）。
5. 见第一节表格：Window 丢早期信息；TokenBuffer 每次算 token 有开销；SummaryBuffer 多一次摘要调用；Entity 成本最高。
6. `assign` 保留原输入并追加字段，扩展性更好；手写 dict 需要在每个调用点重复拼，且容易漏键。
7. `bind` 是调用方不可覆盖的固化参数；`configurable_fields` 是暴露给 `config` 的运行时开关，适合按请求动态调整。
8. `on_end` 不触发，这一轮不会入库；应在 `on_error` 里补记录或告警，或改用 try/finally 手动 `save_context`。

</details>

---

## 十二、本地落地记录与验收（已完成）

### 12.1 变更清单

| 文件 | 变更 |
| --- | --- |
| `../internal/core/memory.py` | **新增**：`FileChatMessageHistory`（JSON 落盘 + 3 轮窗口裁剪）+ `get_file_chat_message_history` 工厂 |
| `../internal/handler/app_handler.py` | `debug()` 改为「三段式 prompt + `RunnableWithMessageHistory` + `session_id=app_id`」 |
| `../.gitignore` | 新增 `storage/`，运行时会话数据不进版本库 |
| `../requirements.txt` | 无需新增：`langchain_core`、`langchain_openai`、`flask_cors` 已在清单中 |
| `../internal/handler/app_handler_1.py` | 保留单轮旧版，仅作对照，不参与运行 |

> 不要执行课程里的 `pip install "langchain~=0.2.1" langchain-community`：会把现有 1.x 降级，并引入与本地代码不兼容的 API。

### 12.2 启动与验收

启动（必须用项目 venv 的解释器，且工作目录是 `llmops-api`）：

```powershell
.\.venv\Scripts\python.exe -m app.http.app
```

确认 `.env` 里有 `OPENAI_API_KEY` 与 `OPENAI_API_BASE`，然后连续两轮请求**同一个** `app_id`：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/apps/<你的app_id>/debug" -Method Post -ContentType "application/json" -Body '{"query":"我叫小明"}'
Invoke-RestMethod -Uri "http://127.0.0.1:5000/apps/<你的app_id>/debug" -Method Post -ContentType "application/json" -Body '{"query":"我叫什么"}'
```

预期：

1. 第二轮能答出「小明」；
2. `../storage/memory/<app_id>.json` 出现两轮、共 4 条消息（human / ai 成对）；
3. 继续聊到第 4 轮，文件里只剩 6 条（最近 3 轮）；
4. 换一个 `app_id` 请求，模型不知道「小明」——隔离生效。

清空某个应用的记忆：删除 `storage/memory/<app_id>.json`，或调用 `clear()`。

### 12.3 常见报错速查

| 报错 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'langchain.memory'` | 照抄了课程 0.2.x 代码 | 改用 `RunnableWithMessageHistory`（第八节） |
| `TypeError: 'WindowsPath' object is not subscriptable` | 写成 `Path(__file__).parent[2]` | 改成 `.parents[2]` |
| `AttributeError: 'FileChatMessageHistory' object has no attribute 'messages'` | 方法名写成单数 `message`，或漏了 `@property` | 改成 `messages` / `add_messages` |
| `KeyError: 'chat_history'` | 占位符名与 `history_messages_key` 不一致 | 两边统一为 `history` |
| 控制台 `LangChainDeprecationWarning` | `RunnableWithMessageHistory` 已过时 | 忽略；等后续章节换 LangGraph |
