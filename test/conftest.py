#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : conftest.py
@Desc    : pytest全局fixture，所有测试目录自动生效
"""
import pytest

from app.http.app import app


@pytest.fixture
def client():
    """获取Flask应用的测试客户端，并自动管理请求上下文"""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
