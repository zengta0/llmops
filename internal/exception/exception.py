from pkg.response import HttpCode
from typing import Any

class CustomException(Exception):
    code:HttpCode = HttpCode.FAIL
    message: str = ""
    data: Any = None

    def __init__(self, message: str = None, data:Any = None):
        super().__init__()
        self.message = message
        self.data = data

class FailException(CustomException):
    """通常失败异常"""
    pass

class NotFoundException(CustomException):
    """未找到数据异常"""
    code = HttpCode.NOT_FOUND


class UnauthorizedException(CustomException):
    """未授权异常"""
    code = HttpCode.UNAUTHORIZED


class ForbiddenException(CustomException):
    """无权限异常"""
    code = HttpCode.FORBIDDEN


class ValidateErrorException(CustomException):
    """数据验证异常"""
    code = HttpCode.VALIDATE_ERROR