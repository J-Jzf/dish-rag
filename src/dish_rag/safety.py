"""拒答规则和用户限制条件保护规则。"""

from dataclasses import dataclass


REFUSAL_KEYWORDS = (
    "生吃河豚",
    "毒蘑菇",
    "非法药物",
    "治疗",
    "治愈",
)

CONSTRAINT_KEYWORDS = (
    "过敏",
    "不要",
    "不能",
    "不吃",
    "无",
    "低糖",
    "少糖",
    "少油",
    "不辣",
    "微辣",
    "清真",
    "素食",
    "纯素",
    "孕妇",
    "儿童",
)


@dataclass(frozen=True)
class SafetyDecision:
    """拒答检查器返回的小型结果对象。"""

    allowed: bool
    reason: str = ""


def check_refusal(query: str) -> SafetyDecision:
    """提前拒绝不安全或医疗化的做菜请求。"""

    normalized = query.strip()
    for keyword in REFUSAL_KEYWORDS:
        if keyword in normalized:
            return SafetyDecision(
                allowed=False,
                reason=f"请求包含高风险或医疗化内容：{keyword}",
            )
    return SafetyDecision(allowed=True)


def extract_user_constraints(query: str) -> list[str]:
    """抽取 Query 重写时必须保留的显式饮食限制。"""

    constraints: list[str] = []
    clauses = query.replace("，", "。").replace(",", "。").split("。")
    for clause in clauses:
        if any(keyword in clause for keyword in CONSTRAINT_KEYWORDS):
            constraints.append(clause.strip())
    return [constraint for constraint in constraints if constraint]


def detect_constraint_loss(original_constraints: list[str], rewritten_query: str) -> list[str]:
    """找出在重写 Query 中消失的限制条件。"""

    missing: list[str] = []
    for constraint in original_constraints:
        # 这里刻意使用保守的子串检查。如果模型把安全或过敏限制改写得太自由，
        # 图应该重试或向用户澄清，而不是悄悄削弱用户的原始要求。
        if constraint and constraint not in rewritten_query:
            missing.append(constraint)
    return missing
