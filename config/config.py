# =============================================================================
# config/config.py —— 配置加载层
#
# 职责：把「环境变量（.env 文件 / 系统环境变量）」翻译成 Python 对象上的属性。
# 取值优先级：环境变量 > DEFAULT_CONFIG 里的默认值。
#
# 为什么要把读取逻辑写在 __init__ 里，而不是写成类属性？
#   类属性（直接写在 class 下）会在 import 这个模块时就求值；
#   而 .env 是在 app/http/app.py 里通过 dotenv.load_dotenv() 注入 os.environ 的。
#   如果写成类属性，import 发生在 load_dotenv 之前，读到的全是 None。
#   写在 __init__ 里，则只有真正 Config() 实例化（load_dotenv 之后）才读，
#   顺序就安全了。这也是 app.py 里能先 import 再 load_dotenv 的原因。
# =============================================================================

import os
from typing import Any

from .default_config import DEFAULT_CONFIG


def _get_env(key: str) -> Any:
    """从环境变量中读取配置项，读不到时回退到 DEFAULT_CONFIG 的默认值。

    os.getenv(key, default) 的第二个参数只在「key 完全不存在」时生效；
    如果 key 存在但值是空字符串 ""，返回的仍是 ""（不会走默认值）。

    返回值标注为 Any，是因为默认值里既有字符串（"False"）也有整数（30），
    类型注解在运行时并不做检查，只起提示作用。
    """
    return os.getenv(key, DEFAULT_CONFIG.get(key))


def _get_bool_env(key: str) -> Any:
    """从环境变量中读取布尔型配置项。

    环境变量本质上都是字符串，所以不能直接 bool("False")（结果会是 True）,
    这里的判定规则是「转小写后是否等于 "true"」：

        "True" / "TRUE" / "true"  → True
        "False" / "0" / "1" / "yes" / "on" → False   ← 只认 true，不认这些写法

    取不到值（None）时也返回 False。
    """
    value: str = _get_env(key)
    return value.lower() == "true" if value is not None else False


class Config:
    """应用配置类。

    命名注意：Flask 的 app.config.from_object(conf) 只会搬运「大写开头」的属性，
    所以这里所有配置项都必须用大写命名，写成小写会被静默忽略。
    """

    def __init__(self):
        # ---------------------------------------------------------------------
        # 1. WTF_CSRF_ENABLED —— Flask-WTF 的 CSRF 防护总开关
        # ---------------------------------------------------------------------
        # 【是什么】
        #   CSRF = Cross-Site Request Forgery，跨站请求伪造。
        #
        # 【攻击场景】
        #   你已经登录了 A 站（浏览器里保存着 A 站的 Cookie），
        #   此时又打开了恶意网站 B，B 的页面里藏了一个自动提交的表单：
        #       <form action="https://a.com/transfer" method="post">...</form>
        #   浏览器提交这个表单时，会「自动带上 A 站的 Cookie」，
        #   A 站无法分辨这次请求是你本人点的，还是 B 站伪造的。
        #
        # 【防御原理】
        #   服务端在渲染表单时，额外下发一个攻击者猜不到的随机 token：
        #       一份存在服务端 session 里；
        #       一份作为隐藏字段 <input name="csrf_token" value="..."> 放进页面。
        #   提交时两者必须匹配。B 站受「同源策略」限制，读不到 A 站页面里的 token，
        #   所以伪造请求会因 token 缺失或不匹配而被拒绝。
        #
        # 【Flask-WTF 的具体实现】
        #   1) 模板里调用 form.hidden_tag() → 生成 csrf_token 隐藏字段，
        #      值是用 SECRET_KEY 对 session 中的随机值签名得到的字符串；
        #   2) FlaskForm.validate() 的第一步就是 validate_csrf()：
        #        从 request.form["csrf_token"] 取值
        #        → 用 SECRET_KEY 验签、检查是否过期（默认 WTF_CSRF_TIME_LIMIT=3600 秒）
        #        → 与 session 里保存的值比对
        #   3) 校验失败时，form.errors 里会出现
        #        "The CSRF token is missing."（没传）或
        #        "The CSRF token is invalid."（传了但对不上）
        #
        # 【本项目为什么要关闭】
        #   这是一个纯 JSON API 项目：
        #     - 前端不会渲染 HTML 表单，也就不会提交 csrf_token 字段；
        #     - 也没有用 Cookie/Session 维持登录态。
        #   如果保持开启，CompletionReq().validate() 会因为缺少 token 直接失败，
        #   所有 /app/completion 请求都会拿到 CSRF 错误，所以必须关闭。
        #
        # 【关闭后的安全边界】
        #   CSRF 攻击成立的前提是「浏览器会自动携带凭证（Cookie）」。
        #   本项目没有用 Cookie 鉴权，风险很低；如果以后改成 Cookie 登录，
        #   就需要重新开启，或改用 SameSite Cookie + Token 鉴权方案。
        self.WTF_CSRF_ENABLED = _get_bool_env("WTF_CSRF_ENABLED")

        # ---------------------------------------------------------------------
        # 2. 数据库连接串
        # ---------------------------------------------------------------------
        # 格式：dialect+driver://user:password@host:port/dbname?参数
        # 例：postgresql+psycopg://postgres:postgresql@localhost:5432/llmops?client_encoding=utf8
        # 默认值故意设为空字符串，意味着「必须由 .env 提供」，避免误连到别的库。
        self.SQLALCHEMY_DATABASE_URI = _get_env("SQLALCHEMY_DATABASE_URI")

        # ---------------------------------------------------------------------
        # 3. 连接池参数
        # ---------------------------------------------------------------------
        # Flask-SQLAlchemy 3.x 起，引擎参数统一放在 SQLALCHEMY_ENGINE_OPTIONS 里，
        # 而不是像 2.x 那样写成 SQLALCHEMY_POOL_SIZE 等独立配置项。
        #   pool_size    ：连接池中常驻的连接数（本项目默认 30）
        #   pool_recycle ：连接存活秒数，超过就重建；
        #                  用来避免数据库端已断开、客户端却拿到「死连接」的问题
        # int(...) 是必要的：环境变量读出来是字符串 "30"，而 SQLAlchemy 要求整数；
        # 若没配环境变量，默认值本身就是 int，int(30) 同样成立。
        self.SQLALCHEMY_ENGINE_OPTIONS = {
            "pool_size": int(_get_env("SQLALCHEMY_POOL_SIZE")),
            "pool_recycle": int(_get_env("SQLALCHEMY_POOL_RECYCLE")),
        }

        # ---------------------------------------------------------------------
        # 4. 是否把生成的 SQL 打印到控制台
        # ---------------------------------------------------------------------
        # 开启后每执行一次查询/写入都会在终端打印 SQL、参数和耗时，方便调试和学习；
        # 生产环境应关闭，否则日志量巨大且可能泄露数据。
        self.SQLALCHEMY_ECHO = _get_bool_env("SQLALCHEMY_ECHO")
