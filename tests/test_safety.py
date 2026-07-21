from dish_rag.safety import detect_constraint_loss, extract_user_constraints


def test_extract_constraints_keeps_allergy_and_spicy_limits():
    query = "我想做宫保鸡丁，但不要花生，而且不辣。"
    constraints = extract_user_constraints(query)

    assert "不要花生" in constraints[0]
    assert "不辣" in constraints[1]


def test_detect_constraint_loss_flags_missing_rewrite_terms():
    constraints = ["不要花生", "不辣"]
    missing = detect_constraint_loss(constraints, "宫保鸡丁 做法 少辣")

    assert missing == ["不要花生", "不辣"]


def test_extract_constraints_without_punctuation_or_with_spaces():
    query = "宫保鸡丁 不要花生 不辣"
    constraints = extract_user_constraints(query)

    assert "不要花生" in constraints
    assert "不辣" in constraints


def test_extract_constraints_without_any_separator():
    query = "宫保鸡丁不要花生不辣"
    constraints = extract_user_constraints(query)

    assert "不要花生" in constraints
    assert "不辣" in constraints
