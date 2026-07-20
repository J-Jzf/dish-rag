from dish_rag.ingest.chunker import chunk_recipe
from dish_rag.models import Recipe


def test_chunker_splits_steps_as_individual_chunks():
    recipe = Recipe(
        recipe_id="001",
        name="宫保鸡丁",
        page_start=3,
        page_end=3,
        cuisine="川菜",
        category="热菜",
        cooking_method="炒",
        difficulty="中等",
        time="约 35 分钟",
        ingredients=["鸡腿肉 300g", "花生米 50g"],
        steps=["切丁", "调汁", "炒熟"],
    )

    chunks = chunk_recipe(recipe)
    step_chunks = [chunk for chunk in chunks if chunk.step_no]

    assert len(step_chunks) == 3
    assert step_chunks[0].chunk_id == "001:step_01"
