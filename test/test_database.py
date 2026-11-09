#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : test_database.py
@Desc    : 验证数据库连接与基础配置是否生效
"""
import pytest
from dotenv import load_dotenv
from flask import Flask
from sqlalchemy import inspect, text

# 先加载.env，保证Config()能读到真实配置
load_dotenv()

from config import Config
from internal.extension.database_extension import db
from internal.model import App


@pytest.fixture(scope="module")
def app():
    """构建一个仅用于测试的Flask应用，并初始化数据库扩展"""
    flask_app = Flask(__name__)
    flask_app.config.from_object(Config())
    db.init_app(flask_app)
    with flask_app.app_context():
        yield flask_app


def test_database_connection(app):
    """能执行最简单的查询，说明TCP连接、认证、库选择都成功"""
    result = db.session.execute(text("select 1")).scalar()
    assert result == 1


def test_database_name(app):
    """确认实际连上的是llmops库，而不是默认的postgres库"""
    name = db.session.execute(text("select current_database()")).scalar()
    assert name == "llmops"


def test_tables(app):
    """打印当前库里的表，不强制要求表已存在"""
    tables = inspect(db.engine).get_table_names()
    print(f"\n当前数据库中的表: {tables}")
    assert isinstance(tables, list)


def test_model_metadata():
    """模型已注册到元数据里（这一步不依赖数据库连接）"""
    assert "app" in db.metadata.tables
    assert App.__tablename__ == "app"
