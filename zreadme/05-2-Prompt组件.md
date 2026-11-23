# 05-2 Prompt 组件

> 返回总览：[05-LangChain核心组件详解.md](./05-LangChain核心组件详解.md)｜上一节：[05-1-消息体系.md](./05-1-消息体系.md)｜下一节：[05-3-Model组件.md](./05-3-Model组件.md)
> 本节重点：模板、占位符、拼接、复用；以及不用 LangChain 时这些都退化成什么。

## 一、LangChain 的 Prompt 组件

### 1.1 三种取结果方式

先看最基础的模板用法：

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

### 1.2 ChatPromptTemplate + MessagesPlaceholder + partial

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

### 1.3 拼接：字符串 `+` 与模板 `+`

字符串模板也能用 `+` 拼接：

```python
prompt = (
        PromptTemplate.from_template("请讲一个关于{subject}的冷笑话")
        + ",让我开心下" +
        "\n使用{language}语言"
)
print(prompt.invoke({"subject": "程序员", "language": "中文"}).to_string())
```

`PromptTemplate + str` 走 `__add__`；`str + PromptTemplate` 走 `__radd__`。结果仍是**模板对象**，`input_variables` 是并集 `{"subject","language"}`。别当普通字符串拼接——先 `format` 再 `+` 会丢变量。

两个消息模板相加：

```python
chat_prompt = system_chat_prompt + human_chat_prompt
```

两个 ChatPromptTemplate 相加 = **消息列表拼接**（左前右后）。价值是把「人设」「历史」「用户输入」拆成独立片段按需组合。

### 1.4 复用：PipelinePromptTemplate

把公共骨架拆成子模板复用：

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

## 二、不用 LangChain 时：提示词就是字符串拼接

最原始的写法是 f-string，system 人设、历史、本轮提问全靠手写：

```python
def build_prompt(query: str, now: str, history: list[tuple[str, str]]) -> str:
    lines = [f"你是OpenAI开发的聊天机器人，请根据用户的提问进行回复，当前的时间为:{now}"]
    for role, content in history:
        lines.append(f"{role}: {content}")
    lines.append(f"Human: 请讲一个关于{query}的冷笑话")
    return "\n".join(lines)


prompt_text = build_prompt(
    "程序员",
    "2024-06-08 21:36:00",
    [("Human", "我叫慕小课"), ("AI", "你好，我是ChatGPT，有什么可以帮到您")],
)
```

如果想保留一点“模板”味道，可以退一步用标准库的 `str.format`：

```python
TEMPLATE = "请讲一个关于{subject}的冷笑话，使用{language}语言"
prompt_text = TEMPLATE.format(subject="程序员", language="中文")
```

「预填固定变量」在标准库里对应 `functools.partial`：

```python
from functools import partial

joke = partial(TEMPLATE.format, language="中文")
prompt_text = joke(subject="程序员")   # 相当于 LangChain 的 .partial(language="中文")
```

逐项对照：

| 能力 | LangChain | 手写（f-string / `str.format`） |
| --- | --- | --- |
| 变量声明 | `{subject}` 由模板解析，`prompt.input_variables` 可查 | f-string 的变量名只有 IDE 知道，运行时无法列出“需要哪些变量” |
| 漏传变量 | `invoke` 立刻 `KeyError: 'subject'` | f-string 无此概念；`str.format` 漏传才 `KeyError` |
| 输出结构 | `PromptValue`，`to_messages()` 保留 system/human 分界 | 只能产出 `str`，system 只是开头一行文字 |
| 拼接 | `+` 拼模板，`input_variables` 取并集，拼完仍是模板 | `+` 只能拼字符串，拼完变量已替换、无法再填 |
| 固定变量 | `.partial(now=...)` 返回新模板 | `functools.partial` / 闭包 / 默认参数 |
| 骨架复用 | `PipelinePromptTemplate` | 函数嵌套调用，或自己写一套字典配置 |

手写最大的问题不是“麻烦”，而是**丢了结构**：`System:` 只是普通文本，不像真正的 system 消息那样被模型当作最高优先级指令。这也是为什么 LCEL 里要一路传 `PromptValue` 而不是 `str`。
