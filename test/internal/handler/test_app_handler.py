#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : test_app_handler.py
@Desc    : 测试internal/handler/app_handler.py中的接口逻辑
"""
import os
import re

import pytest

from pkg.response import HttpCode


class TestAppHandler:
    """app控制器的测试类"""

    def test_ping(self, client):
        """ping接口主动抛出FailException，验证全局异常处理器"""
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert resp.json.get("code") == HttpCode.FAIL
        assert resp.json.get("message") == "数据未找到"

    @pytest.mark.parametrize("json_data", [{}, {"query": ""}, {"query": "a" * 2001}])
    def test_completion_validate_error(self, json_data, client):
        """缺少query、为空、超过2000字符时都应返回validate_error"""
        resp = client.post("/app/completion", json=json_data)
        assert resp.status_code == 200
        assert resp.json.get("code") == HttpCode.VALIDATE_ERROR
        assert "query" in resp.json.get("data")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="未配置OPENAI_API_KEY，跳过真实请求OpenAI的用例",
    )
    def test_completion(self, client):
        """配置了OpenAI密钥后，校验正常提问能拿到回答"""
        resp = client.post("/app/completion", json={"query": "你好，你是谁?"})
        assert resp.status_code == 200
        assert resp.json.get("code") == HttpCode.SUCCESS
        assert resp.json.get("data").get("content")

    def test_app_crud(self, client):
        """走一遍创建、查询、修改、删除的完整链路"""
        # 1.创建
        resp = client.post("/app")
        assert resp.status_code == 200
        assert resp.json.get("code") == HttpCode.SUCCESS
        app_id = re.search(r"[0-9a-f\-]{36}", resp.json.get("message")).group()
        assert app_id is not None

        # 2.查询
        resp = client.get(f"/app/{app_id}")
        assert resp.json.get("code") == HttpCode.SUCCESS
        assert "测试机器人" in resp.json.get("message")

        # 3.修改
        resp = client.post(f"/app/{app_id}")
        assert resp.json.get("code") == HttpCode.SUCCESS
        assert "zt聊天机器人" in resp.json.get("message")

        # 4.删除
        resp = client.post(f"/app/{app_id}/delete")
        assert resp.json.get("code") == HttpCode.SUCCESS
