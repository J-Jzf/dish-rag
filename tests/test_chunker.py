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
    assert step_chunks[0].text == "第1步：切丁"
    assert step_chunks[1].text == "第2步：调汁"
    assert recipe.steps[0] == "切丁"
    assert step_chunks[0].metadata["previous_step_no"] is None
    assert step_chunks[0].metadata["next_step_no"] == 2
    assert step_chunks[1].metadata["previous_step_no"] == 1
    assert step_chunks[1].metadata["next_step_no"] == 3
    assert step_chunks[2].metadata["next_step_no"] is None
    assert step_chunks[2].metadata["total_steps"] == 3
