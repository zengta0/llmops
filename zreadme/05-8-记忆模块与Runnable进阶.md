# 05-8 记忆模块与 Runnable 进阶

> 返回总览：[05-LangChain核心组件详解.md](./05-LangChain核心组件详解.md)｜上一节：[05-7-落地与换厂商.md](./05-7-落地与换厂商.md)
> 参考源码：`第4周\第2章 LLMOps记忆模块解析与Runnable深入学习\llmops-api\study\7~19`
> 本节重点：模型为什么「没有记忆」、三种记忆策略、`RunnableWithMessageHistory`，以及 `bind` / `configurable_*` / `with_retry` / `with_fallbacks` / `with_listeners`。
> 本地状态：本节内容**尚未迁移**到本地项目，`debug()` 仍是单轮版本；先读懂，再按文末「迁移步骤」动手。

## 读完你能做什么

- 说清「模型本身无状态」是怎么回事，历史到底存在哪、怎么进 prompt；
- 不用 LangChain，手写一版「摘要缓冲记忆」；
- 区分 `ChatMessageHistory` / `BaseChatMemory` / 具体 Memory 三者职责；
- 按场景选 Buffer / TokenBuffer / SummaryBuffer / Entity 四种记忆；
- 用 `RunnableWithMessageHistory` 按 `session_id` 隔离多会话；
- 用 Runnable 进阶方法做「固参、热改配置、热换模型、重试、降级、生命周期钩子」；
- 逐行读懂课程版 `debug()` 的记忆链路，并知道它的局限。

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

---

## 四、BaseChatMemory：把「历史」变成「prompt 变量」

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

## 八、Runnable 进阶六件套（study 15~19）

这些方法都返回新的 Runnable，可以继续 `|`。

### 8.1 `bind`：把参数固化在链里（study 15）

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

### 8.2 `configurable_fields`：运行时改链内部字段（study 16）

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

### 8.3 `configurable_alternatives`：运行时换模型（study 17）

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

### 8.4 `with_retry`：失败重试（study 18-1）

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

### 8.5 `with_fallbacks`：主模型挂了自动降级（study 18-2）

```python
llm = ChatOpenAI(model="gpt-3.5-turbo-18k").with_fallbacks([QianfanChatEndpoint()])
chain = prompt | llm | StrOutputParser()
chain.invoke({"query": "你好，你是?"})   # 主模型报错 → 自动用备用模型
```

手写对照是 `try: main() except: backup()`，`with_fallbacks` 的价值在于**链不用改**，降级关系挂在模型对象上。

### 8.6 `with_listeners`：生命周期钩子（study 19）

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

## 九、常见坑

**坑 1：占位符名字对不上**。`load_memory_variables` 返回 `{"history": ...}`，模板写 `MessagesPlaceholder("chat_history")`，运行时 `KeyError: chat_history`。

**坑 2：`return_messages=False` 却用 MessagesPlaceholder**。字符串不能当消息列表，报类型错误。想用占位符就必须 `return_messages=True`；反过来，模板里用 `{history}` 文本插值时才需要字符串。

**坑 3：`input_key` / `output_key` 与链的键不一致**。`memory.save_context` 读不到 `query` / `output`，直接 `KeyError`。课程版里 `output_key="output"` 能对上，是因为 `StrOutputParser` 的输出被包装成 `{"output": ...}`——如果用自定义输出结构，要跟着改。

**坑 4：把 `on_end` 当 `finally` 用**。异常时 `on_end` 不执行，记忆不会保存，看起来像「这轮白聊了」。

**坑 5：`FileChatMessageHistory` 固定路径 + 多用户**。课程版 `./storage/memory/chat_history.txt` 是全局单文件，两个用户同时聊会串话。工程上至少按 `user_id` 分文件，或用 `RunnableWithMessageHistory` + `session_id`。

**坑 6：`SummaryBuffer` 的隐形成本**。摘要由额外的 LLM 调用生成，`max_token_limit` 设得太小会频繁摘要，费用反而上升。

**坑 7：`configurable_fields` 的 `id` 拼错**。不报错、不生效，表现为「配置传了但结果没变」。排查：`llm.config_specs` 看已注册的 id。

**坑 8：`with_retry` 重试非幂等操作**。写数据库、扣费之类的步骤不要整体包 `with_retry`，否则可能重复执行。

---

## 十、自测题（8 道）

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

## 十一、迁移到本地项目的最小步骤

1. 补依赖（本地 `requirements.txt` 目前缺 langchain）：

   ```bash
   pip install "langchain~=0.2.1" langchain-community
   ```

   如果只是想用 core 能力，至少装 `langchain-core`、`langchain-openai`；`FileChatMessageHistory` 在 `langchain-community` 里。

2. 建目录：`storage/memory/`（并加进 `.gitignore`，聊天记录不该提交）。

3. 改 `../internal/handler/app_handler_1.py` 的 `debug()`：套用第七节代码，注意把模型名换回 `deepseek-flash`，`base_url` 仍走 `.env` 的 `OPENAI_API_BASE`。

4. 验收：连续两轮请求同一 `app_id`——

   ```bash
   curl -X POST http://127.0.0.1:5000/apps/<uuid>/debug -H "Content-Type: application/json" -d "{\"query\":\"我叫慕小课\"}"
   curl -X POST http://127.0.0.1:5000/apps/<uuid>/debug -H "Content-Type: application/json" -d "{\"query\":\"我叫什么？\"}"
   ```

   第二轮能答出「慕小课」，且 `storage/memory/chat_history.txt` 里出现两轮 JSON，即迁移成功。

5. 再进一步：把固定文件路径换成 `RunnableWithMessageHistory` + `session_id`，为后面多用户章节打底。
