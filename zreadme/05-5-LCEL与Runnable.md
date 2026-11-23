# 05-5 LCEL 与 Runnable

> 返回总览：[05-LangChain核心组件详解.md](./05-LangChain核心组件详解.md)｜上一节：[05-4-OutputParser.md](./05-4-OutputParser.md)｜下一节：[05-6-回调与可观测.md](./05-6-回调与可观测.md)
> 本节重点：链的本质、`|` 的原理、并行与 RAG 骨架；以及手写版只能做到哪一步。

## 一、LangChain 的 LCEL 与 Runnable

### 1.1 手写 Chain：链的本质是循环

先手写一版最简 Chain —— 链的本质只是一个循环：

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

### 1.2 `|` 是运算符重载

换成 `|` 之后：

```python
chain = prompt | llm | parser
print(chain.invoke({"query": "请讲一个程序员的冷笑话"}))
```

```python
a | b  ==  RunnableSequence(first=a, last=b)
```

`|` 调用 `Runnable.__or__`；`a | b | c` 左结合成 `RunnableSequence(a, b, c)`。序列自身也是 Runnable，可继续接。

### 1.3 数据流与链上三种调用

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

### 1.4 RunnableParallel：一份输入多个出口

一份输入同时跑两条链：

```python
joke_chain = joke_prompt | llm | parser
poem_chain = poem_prompt | llm | parser

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)
res = map_chain.invoke({"subject": "程序员"})
print(res)   # {"joke": "...", "poem": "..."}
```

- 同一份输入**广播**给每个子链，子链**并行**执行，输出 dict（键就是构造时的键名）；
- `RunnableParallel(a=..., b=...)` 与 `RunnableParallel({"a": ..., "b": ...})` 等价。

### 1.5 模拟检索：RAG 最小雏形

模拟 RAG 的最小骨架：

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

### 1.6 RunnablePassthrough.assign

用 `assign` 简化上面那版：

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

## 二、不用 LangChain 时：顺序调用 + 自己写并发

上面 1.1 那个 `Chain` 类其实就是「不用 LangChain 的编排」：只要每步有统一的 `invoke`，一个 for 循环就能串起来。差别在于手写的 `Chain` 只支持 `invoke`，想要 `batch` / `stream` / `config` 得自己再写一套。

**① 最直白的顺序写法（不用任何自定义类）**

```python
prompt_value = prompt.invoke({"query": "你好，你是?"})   # 若不用 LangChain，这里是 build_prompt(...)
resp = client.chat.completions.create(model="deepseek-flash", messages=prompt_value)
content = resp.choices[0].message.content or ""
```

**② 一份输入两个出口（对应 `RunnableParallel`）**

```python
from concurrent.futures import ThreadPoolExecutor

def run(prompt_text: str) -> str:
    r = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt_text}],
    )
    return r.choices[0].message.content or ""

inputs = {"joke": "请讲一个关于程序员的冷笑话", "poem": "请写一首关于程序员的小诗"}
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = {key: pool.submit(run, text) for key, text in inputs.items()}
    res = {key: f.result() for key, f in futures.items()}
```

**③ RAG（对应 1.5 / 1.6 的链）**

```python
def rag_answer(query: str) -> str:
    context = retrieval(query)                     # 1. 检索（手写只能显式写）
    messages = [
        {"role": "system", "content": "请根据上下文回答"},
        {"role": "user", "content": f"<context>\n{context}\n</context>\n\n用户的提问是: {query}"},
    ]
    resp = client.chat.completions.create(model="deepseek-flash", messages=messages)
    return resp.choices[0].message.content or ""
```

逐项对照：

| 能力 | LangChain | 手写 openai SDK |
| --- | --- | --- |
| 串联 | `prompt \| llm \| parser`，任意 Runnable 可组合 | for 循环或直接顺序写死，复用性差 |
| 组输入 | dict 字面量 / `itemgetter` / `lambda` 自动转 Runnable | 手写变量与函数，谁传给谁靠人记 |
| 并行 | `RunnableParallel` / `batch` 自动并行 | `ThreadPoolExecutor` + `submit` / `map` |
| 流式 | `chain.stream(x)` 全链逐块 | 只有最外层 `create(stream=True)`，中间步骤无统一协议 |
| 追加字段 | `RunnablePassthrough.assign` 保留原输入 | 手写 dict 拷贝 `{**data, "context": ...}` |
| 三种调用一致 | 每个 Runnable 都自带 invoke/batch/stream | 每个函数各写一份，接口不统一 |

所以「LCEL 的价值」不是让链能跑（手写 for 循环也能跑），而是让**组合、并行、流式、观测**这四件事用同一套协议复用。
