# 05-4 OutputParser

> 返回总览：[05-LangChain核心组件详解.md](./05-LangChain核心组件详解.md)｜上一节：[05-3-Model组件.md](./05-3-Model组件.md)｜下一节：[05-5-LCEL与Runnable.md](./05-5-LCEL与Runnable.md)
> 本节重点：`AIMessage` → `str` / `dict`；以及手写时这层工作长什么样。

## 一、LangChain 的 OutputParser

### 1.1 StrOutputParser

把 `AIMessage` 降成字符串：

```python
parser = StrOutputParser()
content = parser.invoke(llm.invoke(prompt.invoke({"query": "你好，你是?"})))
print(content)
```

职责：`AIMessage` → `.content`。也接受普通字符串（幂等）和 `{"text": "..."}`（取 `text`）。在链尾把 `AIMessage` 降维成 `str`，接口才能直接塞进 `{"content": content}`。

### 1.2 JsonOutputParser

让模型按指定结构输出 JSON：

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

### 1.3 解析失败的常见原因

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `Invalid json output` | 没注入 `format_instructions` | 用 `partial` 注入 |
| 外面包了 ```` ```json ```` | 模型加了 Markdown | `JsonOutputParser` 一般能容忍；自定义 parser 先剥壳 |
| 前后有解释文字 | 约束不足 | 提示词强调「只输出 JSON」 |
| JSON 被截断 | `max_tokens` 太小 | 调大或缩短输出 |
| 字段名不符 | 模型自由发挥 | `Field(description=...)` 写清语义 |
| 引号未转义 | 生成非法 JSON | 强调转义或换更稳的模型 |

## 二、不用 LangChain 时：自己从 `choices` 里抠、自己 `json.loads`

**① 取纯文本**

```python
content = resp.choices[0].message.content or ""   # or "" 处理 None
```

**② 取 JSON：格式说明要自己写进提示词**

```python
import json

FORMAT_INSTRUCTIONS = """请只输出一个 JSON 对象，不要输出任何解释，结构如下：
{"joke": "回答用户的冷笑话", "punchline": "这个冷笑话的笑点"}"""

messages = [
    {"role": "system", "content": FORMAT_INSTRUCTIONS},
    {"role": "user", "content": "请讲一个关于程序员的冷笑话"},
]
resp = client.chat.completions.create(model="deepseek-flash", messages=messages)

raw = (resp.choices[0].message.content or "").strip()
# 模型有时会套一层 ```json ... ```，得自己剥壳
if raw.startswith("```"):
    raw = raw.strip("`")
    if raw.startswith("json"):
        raw = raw[4:]
try:
    joke = json.loads(raw)        # dict
except json.JSONDecodeError:
    joke = {"joke": raw, "punchline": ""}   # 兜底：解析失败保留原文
```

逐项对照：

| 能力 | LangChain | 手写 openai SDK |
| --- | --- | --- |
| 取正文 | `StrOutputParser()` 内部读 `.content` | `resp.choices[0].message.content or ""` |
| 幂等/兼容 dict 输入 | `StrOutputParser` 支持 `str`、`{"text":...}` | 无，输入类型全靠自己约定 |
| 格式说明 | `parser.get_format_instructions()` 按 schema 自动生成 | 手写一段 schema 文本，容易与代码不同步 |
| 结构定义 | `pydantic_object=Joke` 声明字段与语义 | 只在提示词文字里描述，无类型约束 |
| 剥 Markdown 壳 | `JsonOutputParser` 一般能容忍 | 自己判断 ``` 并截取 |
| 解析异常 | parser 抛 `OutputParserException`，链上可捕获 | `json.JSONDecodeError`，自己 try/except |
| 多字段校验 | 交由 Pydantic 模型 | 自己写 `if "joke" not in data` 之类检查 |

要注意：手写版最脆的地方是「格式说明」和「解析代码」是两份独立的东西，改了字段往往只改了一边；`JsonOutputParser + pydantic_object` 的价值就在于让二者共用同一份结构定义。
