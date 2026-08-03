from dish_rag.ingest.parser import parse_recipes_from_pages


def test_parse_recipe_fields_from_page_text():
    page = """
001 宫保鸡丁
基础信息    川菜｜热菜｜炒｜中等｜约 35 分钟｜家庭份量 2-3 人

原材料     去骨鸡腿肉 300g；熟花生米 50g；干辣椒 8 个

详细做法    ①鸡腿肉切丁；
        ②调碗汁；
        ③热锅下油炒鸡丁；
        ④倒入碗汁收浓；
        ⑤拌入花生米

口味特点    咸鲜微辣

适合人群    儿童及胃肠敏感者应减辣

饮食标签    含添加糖、辛辣

过敏原提示   大豆、花生/坚果、酒精

厨具与替换   菜刀、砧板、基础锅具。鸡肉可换厚豆腐；不吃辣可减辣椒

保存与复热   熟制后 2 小时内冷藏。
"""
    recipes = parse_recipes_from_pages([page])
    recipe = recipes[0]

    assert recipe.recipe_id == "001"
    assert recipe.name == "宫保鸡丁"
    assert recipe.cuisine == "川菜"
    assert recipe.category == "热菜"
    assert len(recipe.ingredients) == 3
    assert len(recipe.steps) == 5
    assert "花生/坚果" in recipe.allergens
    assert "不吃辣" in recipe.substitutions


def test_low_confidence_marks_missing_fields():
    page = """
002 简版菜
基础信息    家常菜｜热菜｜炒｜简单｜约 10 分钟
原材料     青菜 200g
详细做法    ①炒熟
"""
    recipe = parse_recipes_from_pages([page])[0]

    assert recipe.parse_confidence < 0.95
    assert recipe.parse_warnings
