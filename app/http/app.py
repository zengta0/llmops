# =============================================================================
# app/http/app.py —— 项目启动入口
#
# 这个文件只做一件事：把「配置、数据库、路由、控制器」这些零件组装成一个 Flask 应用。
#
# 重要：第 9 行用的是相对导入（from .module import ...），
#      所以不能直接 `python app/http/app.py` 运行，否则会报
#          ImportError: attempted relative import with no known parent package
#      正确启动方式：
#          python -m app.http.app
#      或在 PyCharm 运行配置里把 Target 选为 Module name = app.http.app
# =============================================================================

# -----------------------------------------------------------------------------
# 一、导入第三方库
# -----------------------------------------------------------------------------

# import dotenv 导入的是第三方库 python-dotenv：
#   pip 安装名：python-dotenv
#   import 名：dotenv（名字不一致，这是很多初学者踩坑的地方）
# 它的职责：把 .env 文件里的 KEY=VALUE 读进 os.environ（进程环境变量）。
# 注意：只 import 不会读文件，真正读取要靠下面的 dotenv.load_dotenv()。
import dotenv

# flask_migrate 是 Flask-Migrate 库，内部封装了 Alembic（数据库迁移工具）。
# 这里导入的 Migrate 是一个「扩展类」，可以把它理解成数据库迁移功能的开关。
# 项目中真正的 migrate 实例在 internal/extension/migrate_extension.py 里创建：
#     migrate = Migrate()
# 这里之所以还要 import，是因为下面 injector.get(Migrate) 需要这个「类型」作为查询依据。
from flask_migrate import Migrate

# injector 是依赖注入（Dependency Injection，缩写 DI）库。
# 先理解它要解决的问题：
#   Router 需要 AppHandler，AppHandler 需要 AppService，AppService 需要 db，
#   如果全部手写，就要这样一层层 new：
#       service = AppService(db=db)
#       handler = AppHandler(app_service=service)
#       router  = Router(app_handler=handler)
#   对象一多，顺序、传参都容易错，改一个构造函数牵一发而动全身。
# Injector 就是「对象工厂 + 自动装配工」：
#   你只需要声明「什么类型 → 给哪个实例」（写在 module.py 里），
#   剩下的按类型注解递归构造，全部交给容器。
# 这行只是导入类；真正创建容器是下面的 Injector([ExtensionModule])。
from injector import Injector

# -----------------------------------------------------------------------------
# 二、导入项目内部模块
# -----------------------------------------------------------------------------

# config 是本项目自己的包（config/ 目录），Config 类负责把环境变量整理成配置对象。
from config import Config

# internal 是业务核心目录，这里导入两个组件：
#   Router：路由管理器，负责把 URL 和控制器方法（Handler）绑定起来
#   Http  ：Http 服务引擎，继承自 flask.Flask，额外包了配置、扩展、异常处理、路由注册
from internal.router import Router
from internal.server import Http

# pkg 是公共工具目录。pkg/sqlalchemy/sqlalchemy.py 里定义了一个自定义类 SQLAlchemy，
# 它继承自 flask_sqlalchemy 的 SQLAlchemy，并额外增加了 auto_commit() 上下文管理器
# （with db.auto_commit(): 结束时自动提交，出错自动回滚）。
# 注意：这里导入的是「我们自己定义的子类」，不是 flask_sqlalchemy 原生那个同名类。
from pkg.sqlalchemy import SQLAlchemy

# 开头的点 . 表示「相对导入」：从当前包 app.http 里找 module 模块，即 app/http/module.py。
# ExtensionModule 里定义了依赖注入的绑定规则，是下面创建容器的「说明书」。
# 因为用的是相对导入，本文件必须作为 app.http 包的一部分被加载（见文件开头说明）。
from .module import ExtensionModule

# -----------------------------------------------------------------------------
# 三、加载 .env 并生成配置对象
# -----------------------------------------------------------------------------

# load_dotenv() 做了三件事：
#   1. 从当前工作目录向上查找 .env 文件（所以要在项目根目录启动）
#   2. 逐行解析 KEY=VALUE（忽略 # 注释和空行）
#   3. 写进 os.environ；已经存在的同名环境变量默认不覆盖
# 为什么必须放在 Config() 之前？
#   因为 Config.__init__ 里靠 os.getenv(...) 读值，环境变量还没注入的话读到的全是 None。
# 另外注意：上面的 import 语句执行时并不会读配置，只有 Config() 实例化那一刻才读，
# 所以 import 顺序在前、load_dotenv 在后，是没有问题的。
dotenv.load_dotenv()

# Config() 实例化配置类，它内部会读取这些环境变量并挂成实例属性：
#     SQLALCHEMY_DATABASE_URI / SQLALCHEMY_POOL_SIZE / SQLALCHEMY_ECHO
#     WTF_CSRF_ENABLED / OPENAI_API_KEY 等
# 后面 Http 内部会调用 self.config.from_object(conf)，
# 把这些「大写属性」搬进 Flask 的 app.config 字典里（小写属性会被忽略）。
conf = Config()

# -----------------------------------------------------------------------------
# 四、创建依赖注入容器
# -----------------------------------------------------------------------------
# Injector([...]) 的参数是「绑定模块」列表（Module 的子类）。
# app/http/module.py 里 ExtensionModule.configure() 写了两条规则：
#     binder.bind(SQLAlchemy, to=db)        # 谁要 SQLAlchemy，就给全局 db 实例
#     binder.bind(Migrate,    to=migrate)   # 谁要 Migrate，就给全局 migrate 实例
#
# 关键点：这一步只「登记规则」，不会立刻创建 Router/AppHandler/AppService。
# 真正构造对象发生在下一段调用 injector.get(...) 的时候（按需创建，叫懒加载）。
injector = Injector([ExtensionModule])

# -----------------------------------------------------------------------------
# 五、组装并创建 Flask 应用
# -----------------------------------------------------------------------------
# Http 继承自 flask.Flask，它的构造参数除了父类的，还额外要求 4 个组件。
#
# __name__ 是当前模块名：
#     以模块方式运行时是 "app.http.app"
#     Flask 用它定位模板目录、静态文件目录，也用于日志名称。
#
# injector.get(SQLAlchemy)：
#     容器查绑定表 → SQLAlchemy 绑定了 db 实例 → 直接返回 db（全项目共用的单例）
#
# injector.get(Migrate)：
#     同理 → 返回 migrate 实例
#
# injector.get(Router)：
#     Router 没有显式绑定，容器会按构造函数里的类型注解「自动构造」，过程是递归的：
#         Router(app_handler: AppHandler)
#           └→ AppHandler(app_service: AppService)
#                └→ AppService(db: SQLAlchemy)
#                     └→ 命中绑定，拿到全局 db
#     于是 Router / AppHandler / AppService 各被创建一次，并且互相引用好。
#     这就是 @inject + @dataclass + 类型注解 三件套在阶段 5 的兑现时刻。
#
# 进入 Http.__init__ 后，还会依次做这些事（见 internal/server/http.py）：
#     1. super().__init__()          初始化 Flask 本体
#     2. config.from_object(conf)    应用配置
#     3. register_error_handler()    注册全局异常处理器
#     4. db.init_app / migrate.init_app  绑定扩展
#     5. router.register_router(self)    注册所有 URL 规则
app = Http(
    __name__,
    conf=conf,
    db=injector.get(SQLAlchemy),
    migrate=injector.get(Migrate),
    router=injector.get(Router),
)

# -----------------------------------------------------------------------------
# 六、只在「直接运行本入口」时启动开发服务器
# -----------------------------------------------------------------------------
# Python 每个模块都有 __name__：
#     直接执行该文件时        → __name__ == "__main__"
#     被其他模块 import 时    → __name__ 是模块名，例如 "app.http.app"
# 因此下面这段只在主动启动服务时执行；
# 像 `flask db migrate` 那样只是 import 本模块的场景，不会启动服务器，避免端口被占用。
#
# app.run(debug=True) 说明：
#     debug=True：开启调试模式（改代码自动重启、出错显示详细堆栈页面）
#     默认监听 127.0.0.1:5000；
#     想换地址/端口可以写成 app.run(host="0.0.0.0", port=5001, debug=True)
if __name__ == "__main__":
    app.run(debug=True)
