"""应用模块的请求参数定义 —— 基于 FlaskForm（WTForms）。

【这个文件解决什么问题】
接口收到 POST 的 JSON（例如 {"query": "你好"}）后，需要检查 query 是否存在、
长度是否超限。CompletionReq 就是这张“检查表”：字段写在类里，检查规则写在 validators 里。

【先记住一条：括号就是继承】

    class CompletionReq(FlaskForm):

Python 的继承不写 extends，直接把父类放进类名后面的括号里。
等价于 Java / TypeScript 的：class CompletionReq extends FlaskForm
继承链：CompletionReq -> FlaskForm -> wtforms.Form -> BaseForm

【最容易误解的地方：类体里的 query 只是“字段声明”】

    query = StringField("query", validators=[...])

这句不是“把字符串赋给 query”，而是在声明一个表单字段。
StringField(...) 返回的是 UnboundField（还没绑定具体表单的字段模板），
真正的字段对象要等到 CompletionReq() 实例化时才创建出来。

注意：括号里的第一个参数是 label（显示名称），不是 JSON 的 key。
JSON 用哪个 key，是由“类属性名”决定的（属性名 query 对应 JSON 的 query）。

【运行时分两步理解】

第一步：req = CompletionReq() 时，内部按顺序发生：
    1) FlaskForm.__init__ 的参数 formdata 默认值是 _Auto（一个特殊标记对象）。
       为什么不直接用 None？因为 None 在 WTForms 里表示“明确不绑定任何请求数据”，
       而 _Auto 表示“调用方没传，请自动去 request 里找数据”。
    2) 调用 Meta.wrap_formdata()（FlaskForm 自带）判断数据来源：
       - 当前请求方法是否属于 POST / PUT / PATCH / DELETE（也就是“有提交动作”）；
       - 依次尝试 request.files、request.form，最后 request.is_json；
       - 如果请求体是 JSON，就执行 request.get_json()，
         再包装成 ImmutableMultiDict（一种只读的“多值字典”）。
         字段取值只认这种字典结构，所以 JSON 需要先转换一层。
       如果请求方法不是提交类（例如 GET），这里返回 None，所有字段都取不到值。
    3) 每个字段执行 field.process(formdata)：按“类属性名”从上面那份数据中取值，
       存进 field.data。
    4) 用 setattr 把字段对象挂到实例上。
       所以 req.query 不是字符串，而是一个字段对象；
       要拿真正的值必须写 req.query.data。

第二步：req.validate() 时：
    1) 遍历表单中的每个字段，调用 field.validate()；
    2) 先做字段自身检查，再按 validators 列表顺序执行：
       DataRequired 检查“有没有值”，Length(max=2000) 检查“长度是否超限”；
       某条规则失败时错误会追加到 field.errors；DataRequired 失败会立即结束该字段的
       后续规则检查，其他规则失败后仍会继续执行；
    3) 错误信息写入 field.errors，不会抛异常中断程序；
    4) req.errors 只收集有错误的字段，格式为 {字段名: [错误文案]}。

【一个完整例子】

请求：POST /apps/<app_id>/debug，body 为 {"query": "你好"}
    req = CompletionReq()   ->  req.query.data 等于 "你好"
    req.validate()          ->  True

请求 body 为 {}（缺少 query）
    req = CompletionReq()   ->  req.query.data 为 None
    req.validate()          ->  False
    req.errors              ->  {"query": ["用户的提问是必填"]}
这个字典可以直接交给 pkg.response.validate_error_json() 返回给前端。

【多个参数怎么声明（示例）】

一张表单可以声明任意多个字段，每个字段就是一行类属性。
下面是“聊天请求”同时接收 提问、模型名、最大字数、是否流式、难度 的写法：

    from flask_wtf import FlaskForm
    from wtforms import BooleanField, IntegerField, SelectField, StringField
    from wtforms.validators import DataRequired, Length, NumberRange, Optional

    class ChatReq(FlaskForm):
        # 字段名 = 类属性名，JSON 里的 key 必须与属性名一致
        query = StringField("query", validators=[DataRequired(), Length(max=2000)])
        model = StringField("model", validators=[DataRequired(), Length(max=50)])
        max_tokens = IntegerField(
            "max_tokens",
            default=500,
            validators=[NumberRange(min=1, max=4096)],
        )
        stream = BooleanField("stream", default=False)
        level = SelectField(
            "level",
            choices=[("easy", "简单"), ("hard", "困难")],
            validators=[Optional()],
        )

对应的请求 body 可以写成：
    {
        "query": "你好",
        "model": "deepseek-flash",
        "max_tokens": 800,
        "stream": false,
        "level": "easy"
    }

对应取值：
    req.query.data        -> "你好"
    req.model.data        -> "deepseek-flash"
    req.max_tokens.data   -> 800
    req.stream.data       -> False（请求没传时用 default 的 False）
    req.level.data        -> "easy"

多个字段时的几个要点：
    1. JSON 的 key 由类属性名决定；括号里第一个参数只是 label（显示名称）。
       把 "max_tokens" 改成 "最大字数" 只影响显示，取值仍然看属性名 max_tokens。
    2. 字段声明顺序 = validate 的遍历顺序，也是页面上渲染字段的顺序。
    3. 校验会“一次跑完全部字段”，不会遇到第一个错误就停。
       所以 req.errors 可能同时包含多个字段：
       {"query": ["用户的提问是必填"], "max_tokens": ["数值必须介于 1 和 4096 之间"]}
    4. 默认值 default 只在“请求里没传这个字段”时生效；
       它不代表必填，也不代表非必填——是否必填由 validators 里的 DataRequired 决定。
       数组/字典等可变默认值建议写成 lambda，例如 default=lambda: []，
       避免多个请求共用同一个可变对象。
    5. IntegerField 收到非数字（如 "abc"）时不会抛异常，
       而是绑定失败并记录到该字段的 errors，随后正常参与 validate()。
    6. SelectField 只有出现在 choices 里的值才算合法；
       想让它是可选字段，就用 Optional 校验器。

【补充说明】
- 以后表单要加字段，继续按同样方式在类里声明即可，不用手写解析 JSON 的代码。
- CSRF 校验也走字段机制：Meta.csrf 为真时，WTForms 会自动追加 csrf_token 字段。
  本项目在 config 中关闭了 WTF_CSRF_ENABLED，所以纯 JSON 接口不受影响。
- 相关源码位置（venv 为当前项目的虚拟环境）：
    venv/Lib/site-packages/flask_wtf/form.py
    venv/Lib/site-packages/wtforms/form.py
"""

from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired, Length


class CompletionReq(FlaskForm):
    """基础聊天接口请求验证。"""

    # 这里只是“声明字段”：CompletionReq() 实例化时才会生成字段对象，
    # 并从 request.get_json() 中读取同名 key 的值（详见文件顶部说明）。
    # 必填、长度最大为2000
    query = StringField(
        "query",
        validators=[
            DataRequired(message="用户的提问是必填"),
            Length(max=2000, message="用户的提问最大长度是2000"),
        ],
    )