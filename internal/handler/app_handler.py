from internal.service import AppService
from pkg.response import success_json, validate_error_json, success_message
import uuid
from internal.schema.app_schema import CompletionReq
from openai import OpenAI
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from internal.exception import FailException
from dataclasses import dataclass

from injector import inject

@inject
@dataclass
class AppHandler:
    """应用控制器"""
    app_service: AppService

    def create_app(self):
        app = self.app_service.create_app()
        return success_message(f"引用已成功创建， id为{app.id}")

    def get_app(self, id: uuid.UUID):
        app = self.app_service.get_app(id)
        return success_message(f"应用已经成功获取，名字是{app.name}")

    def update_app(self, id: uuid.UUID):
        app = self.app_service.update_app(id)
        return success_message(f"应用已经成功修改，修改的名字是:{app.name}")

    def delete_app(self, id: uuid.UUID):
        app = self.app_service.delete_app(id)
        return success_message(f"应用已经成功删除，id为:{app.id}")

    def completion(self):
        """聊天接口"""
        # 1.提取从接口中获取的输入，POST
        req = CompletionReq()
        if not req.validate():
            return validate_error_json(req.errors)

        # 2.构建OpenAI客户端，并发起请求
        client = OpenAI(base_url=os.getenv("OPENAI_API_BASE"))

        # 3.得到请求响应，然后将OpenAI的响应传递给前端
        completion = client.chat.completions.create(
            model="deepseek-flash",
            messages=[
                {"role": "system", "content": "你是OpenAI开发的聊天机器人，请根据用户的输入回复对应的信息"},
                {"role": "user", "content": req.query.data},
            ]
        )

        content = completion.choices[0].message.content

        return success_json({"content": content})

    def debug(self, app_id: UUID):
        """聊天接口

        参数 app_id 来自路由 /apps/<uuid:app_id>/debug。
        注意：当前版本只用到它的「格式校验」——Flask 的 uuid 转换器会保证它
        是合法 UUID，但方法内部并没有拿它去查数据库，所以不存在的 UUID
        也能正常返回内容。后续章节才会用它读取应用配置、对话记录等。
        """

        # 1.校验并提取请求体中的输入
        # CompletionReq 继承自 FlaskForm，实例化时会自动读取 request.get_json()，
        # 把 JSON 里的字段绑定到 form 属性上；validate() 触发校验规则
        # （例如 query 不能为空、长度不超过 2000）。
        # 校验失败时把「字段 → 错误信息」原样返回给前端，由前端提示用户。
        req = CompletionReq()
        if not req.validate():
            return validate_error_json(req.errors)

        # 2.构建 LangChain 的三个基础组件
        #
        # 2.1 提示词模板（PromptTemplate）
        # from_template 里的 {query} 是「占位变量」，不是 Python 的 f-string。
        # invoke 时会用传入字典中 key 为 query 的值去替换它，
        # 最终拼成发送给模型的完整提示词。变量名必须与 invoke 的字典键一致，
        # 否则运行时会抛 KeyError。
        prompt = ChatPromptTemplate.from_template("{query}")
        #
        # 2.2 聊天模型（ChatModel）
        # ChatOpenAI 是 LangChain 对「OpenAI 兼容接口」的封装，内部仍然是
        # openai SDK 发 HTTP 请求。这里没有传 base_url 和 api_key，
        # 它会按顺序从环境变量解析：
        #     base_url: 显式参数 → OPENAI_API_BASE → OPENAI_BASE_URL → 官方地址
        #     api_key : 显式参数 → OPENAI_API_KEY
        # 这些环境变量由 app/http/app.py 里的 dotenv.load_dotenv() 从 .env 注入。
        # model 参数会原样放进请求体的 model 字段，所以这里可以写 deepseek-flash。
        llm = ChatOpenAI(model="deepseek-flash")
        #
        # 2.3 输出解析器（OutputParser）
        # 模型返回的是 AIMessage 对象（里面还包含 token 用量、结束原因等元数据），
        # StrOutputParser 只取出其中的 .content 文本，转成普通字符串。
        parser = StrOutputParser()

        # 3.用 LCEL 把三个组件串成一条链
        # 「|」不是位运算，而是 LangChain 重载的运算符：a | b 会生成一个
        # RunnableSequence（可运行序列），表示「a 的输出交给 b 作为输入」。
        # 数据流向：dict → PromptValue → AIMessage → str
        # 这样写的好处是：每个组件都是独立、可替换的，换模型或换解析器
        # 只需要改其中一行，不用重写整段逻辑。
        chain = prompt | llm | parser

        # 4.执行链
        # invoke 是同步调用（对应还有 stream 流式、batch 批量）。
        # 传入的字典就是给提示词模板填变量用的；如果模板里还有别的占位符，
        # 必须在这里一并提供。
        content = chain.invoke({"query": req.query.data})
        print('生成内容:', content)

        # 5.按项目统一的响应格式返回
        # success_json 会把 data 包装成 {"code": "success", "data": {...}, "message": ""}，
        # 前端 request.ts 正是靠 code 判断成功与否，再取 data.content 显示。
        return success_json({"content": content})

    def ping(self):
        raise FailException("数据未找到")
        # return {"ping": "pong"}