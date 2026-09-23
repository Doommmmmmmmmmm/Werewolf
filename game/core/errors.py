class RuleViolationError(ValueError):
    """玩家行动不符合当前规则或阶段时抛出。"""

    def __init__(self, message: str, **details: object) -> None:
        super().__init__(message)
        self.details = details


class ModelClientError(RuntimeError):
    """模型服务不可用或返回不可解析内容时抛出。"""

