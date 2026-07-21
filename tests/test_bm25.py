from dish_rag.models import RecipeChunk, RecipeField
from dish_rag.retrieval.bm25 import LocalBM25


def test_local_bm25_respects_recipe_id_filter():
    chunks = [
        RecipeChunk(
            chunk_id="001:step_02",
            recipe_id="001",
            recipe_name="宫保鸡丁",
            field=RecipeField.STEPS,
            text="第2步：调好碗汁",
            page=3,
            step_no=2,
        ),
        RecipeChunk(
            chunk_id="054:step_05",
            recipe_id="054",
            recipe_name="油爆双脆",
            field=RecipeField.STEPS,
            text="第5步：倒入碗汁出锅",
            page=20,
            step_no=5,
        ),
    ]

    hits = LocalBM25(chunks).search("调好碗汁后做什么", filters={"recipe_id": "001"})

    assert hits
    assert {hit.recipe_id for hit in hits} == {"001"}
    assert all(hit.filters == {"recipe_id": "001"} for hit in hits)
