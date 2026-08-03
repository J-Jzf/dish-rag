from dish_rag.agent import prompts


def test_multi_intent_template_formats_json_example_without_key_error():
    rendered = prompts.INTENT_USER.format(
        query="推荐三道高蛋白菜",
        cooking_state="{}",
        memory="{}",
    )

    assert '"actions"' in rendered
    assert "推荐三道高蛋白菜" in rendered
