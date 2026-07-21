"""拒答规则和用户限制条件保护规则。"""

from dataclasses import dataclass
import re


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


@dataclass(frozen=True) # 创建一个不可修改的数据类
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
    """抽取 Query 重写时必须保留的显式饮食限制。输出用户原始query，输出限制条件list。"""

    constraints: list[str] = []
    # 先用常见标点、空白、换行切分；用户不打中文标点但用空格分隔时也能识别。
    clauses = re.split(r"[。，,；;、\s]+", query)
    for clause in clauses:
        normalized_clause = clause.strip()
        if any(keyword in normalized_clause for keyword in CONSTRAINT_KEYWORDS):
            constraints.append(normalized_clause)

    # 再处理完全不打分隔符的情况，例如“宫保鸡丁不要花生不辣”。
    # 这里从每个限制关键词开始，截取到下一个限制关键词或句尾。
    keyword_pattern = "|".join(re.escape(keyword) for keyword in CONSTRAINT_KEYWORDS)
    matches = list(re.finditer(keyword_pattern, query))
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(query)
        fragment = query[start:end].strip(" ，,。；;、\t\r\n")
        if fragment:
            constraints.append(fragment)

    # 去重并保留原始顺序，返回非空限制条件。
    unique_constraints: list[str] = []
    for constraint in constraints:
        if constraint and constraint not in unique_constraints:
            unique_constraints.append(constraint)
    return unique_constraints


def detect_constraint_loss(original_constraints: list[str], rewritten_query: str) -> list[str]:
    """找出在重写 Query 中消失的限制条件（用户query有但重写后无或被弱化了的）。"""

    missing: list[str] = []
    for constraint in original_constraints:
        # 这里刻意使用保守的子串检查。如果模型把安全或过敏限制改写得太自由，
        # 图应该重试或向用户澄清，而不是悄悄削弱用户的原始要求。
        if constraint and constraint not in rewritten_query:
            missing.append(constraint)
    return missing
