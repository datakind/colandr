import pandas as pd
import pytest
import river.compose

from colandr.lib.models import StudyRanker


@pytest.fixture(scope="class")
def tmp_study_ranker_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("ranker_models")
    yield tmp_path


class TestStudyRanker:
    @pytest.mark.parametrize("review_id", [1, 2])
    def test_init(self, review_id, tmp_study_ranker_path):
        sranker = StudyRanker(review_id, tmp_study_ranker_path)
        assert sranker.review_id == review_id
        assert sranker.dir_path == tmp_study_ranker_path
        assert sranker._model is None

    @pytest.mark.parametrize("review_id", [1, 2])
    def test_model_fpath(self, review_id, tmp_study_ranker_path):
        sranker = StudyRanker(review_id, tmp_study_ranker_path)
        assert tmp_study_ranker_path in sranker.model_fpath.parents
        assert f"review_{review_id:06}" in sranker.model_fpath.name

    @pytest.mark.parametrize("review_id", [1, 2])
    def test_dunders(self, review_id, tmp_study_ranker_path):
        sranker = StudyRanker(review_id, tmp_study_ranker_path)
        assert sranker == StudyRanker(review_id, tmp_study_ranker_path)  # __eq__
        assert sranker in {StudyRanker(review_id, tmp_study_ranker_path)}  # __hash__

    @pytest.mark.parametrize("review_id", [1, 2])
    def test_model(self, review_id, tmp_study_ranker_path):
        sranker = StudyRanker(review_id, tmp_study_ranker_path)
        assert sranker.model() is not None
        assert isinstance(sranker.model(), river.compose.Pipeline)

    @pytest.mark.parametrize(
        "record",
        [
            {"text": "Mary had a little lamb.", "target": True},
            {"text": "Its fleece was white as snow.", "target": False},
            {"text": "And everywhere that Mary went...", "target": True},
            {"text": "The lamb was sure to go.", "target": False},
        ],
    )
    def test_learn_one(self, record, tmp_study_ranker_path):
        sranker = StudyRanker(1, tmp_study_ranker_path)
        sranker.learn_one(record)
        model_ = sranker.model()
        assert model_["classifier"].weights
        assert any(value > 0.0 for value in model_["classifier"].weights.values())
        assert (
            "mary" in model_["classifier"].weights
            and model_["classifier"].weights["mary"] > 0.0
        )

    @pytest.mark.parametrize(
        "records",
        [
            (
                {"text": "Mary had a little lamb.", "target": True},
                {"text": "Its fleece was white as snow.", "target": False},
            ),
            (
                {"text": "And everywhere that Mary went...", "target": True},
                {"text": "The lamb was sure to go.", "target": False},
            ),
        ],
    )
    def test_learn_many(self, records, tmp_study_ranker_path):
        sranker = StudyRanker(2, tmp_study_ranker_path)
        sranker.learn_many(records)
        model_ = sranker.model()
        assert model_["classifier"].weights
        assert any(value > 0.0 for value in model_["classifier"].weights.values())
        assert (
            "mary" in model_["classifier"].weights
            and model_["classifier"].weights["mary"] > 0.0
        )

    @pytest.mark.parametrize(
        ["record", "proba", "exp_pred"],
        [
            ({"text": "Mary ate a little breakfast."}, False, True),
            ({"text": "The lamb was white in color."}, True, False),
            ({"text": "Mary went everywhere with the lamb."}, True, True),
            ({"text": "Fleece is soft and fluffy like snow."}, False, False),
        ],
    )
    def test_predict_one(self, record, proba, exp_pred, tmp_study_ranker_path):
        sranker = StudyRanker(1, tmp_study_ranker_path)
        pred = sranker.predict_one(record, proba=proba)
        if proba:
            assert pred and isinstance(pred, dict)
            assert pred[True] >= 0.0 and pred[True] <= 1.0
            assert pred[True] > pred[False] if exp_pred else pred[True] < pred[False]
        else:
            assert isinstance(pred, bool)
            assert pred is exp_pred

    @pytest.mark.parametrize(
        ["records", "proba", "exp_preds"],
        [
            (
                [
                    {"text": "Mary ate a little breakfast."},
                    {"text": "The lamb was white in color."},
                ],
                False,
                pd.Series([True, False]),
            ),
            (
                [
                    {"text": "Mary went everywhere with the lamb."},
                    {"text": "Fleece is soft and fluffy like snow."},
                ],
                True,
                pd.Series([True, False]),
            ),
        ],
    )
    def test_predict_many(self, records, proba, exp_preds, tmp_study_ranker_path):
        sranker = StudyRanker(2, tmp_study_ranker_path)
        preds = sranker.predict_many(records, proba=proba)
        if proba:
            assert isinstance(preds, pd.DataFrame)
            assert (preds.columns == [False, True]).all()
            assert preds.idxmax(axis="columns").equals(exp_preds)
        else:
            assert isinstance(preds, pd.Series)
            assert preds.equals(exp_preds)

    @pytest.mark.parametrize("review_id", [1, 2])
    def test_save(self, review_id, tmp_study_ranker_path):
        sranker1 = StudyRanker(review_id, tmp_study_ranker_path)
        sranker1.save()
        sranker2 = StudyRanker(review_id, tmp_study_ranker_path)
        assert (
            sranker1.model()["classifier"].weights
            == sranker2.model()["classifier"].weights
        )
