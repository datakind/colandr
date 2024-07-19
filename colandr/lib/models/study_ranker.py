import collections
import functools
import logging
import pathlib
from collections.abc import Iterable

import joblib
import pandas as pd
import river.base
import river.compose
import river.datasets
import river.evaluate
import river.feature_extraction
import river.linear_model
import river.metrics
import river.naive_bayes
import river.optim
import river.preprocessing
import scipy.sparse


LOGGER = logging.getLogger(__name__)


class ColandrTFIDF(river.feature_extraction.TFIDF):
    def learn_many(self, X: pd.Series) -> None:
        # increment global document counter
        self.n += X.shape[0]
        # update document counts
        doc_counts = (
            X.map(lambda x: set(self.process_text(x)))
            .explode()
            .value_counts()
            .to_dict()
        )
        self.dfs.update(doc_counts)

    def transform_many(self, X: pd.Series) -> pd.DataFrame:
        """Transform pandas series of string into tf-idf pandas sparse dataframe."""
        indptr, indices, data = [0], [], []
        index: dict[int, int] = {}
        for doc in X:
            term_weights: dict[int, float] = self.transform_one(doc)
            for term, weight in term_weights.items():
                indices.append(index.setdefault(term, len(index)))
                data.append(weight)
            indptr.append(len(data))

        return pd.DataFrame.sparse.from_spmatrix(
            scipy.sparse.csr_matrix((data, indices, indptr)),
            index=X.index,
            columns=index.keys(),
        )


class StudyRanker:
    _model_fname: str = "study_ranker__review_{review_id}.pkl"

    def __init__(
        self,
        *,
        review_id: int,
        dir_path: str | pathlib.Path,
        feature_col: str = "text",
        target_col: str = "target",
    ):
        self.review_id = review_id
        self.dir_path = pathlib.Path(dir_path)
        self.feature_col = feature_col
        self.target_col = target_col
        self.model = _load_study_ranker_model(review_id, self.model_fpath)

    def __str__(self) -> str:
        return (
            "StudyRanker("
            f"review_id={self.review_id}, "
            f"dir_path='{self.dir_path}', "
            f"feature_col='{self.feature_col}', "
            f"target_col='{self.target_col}'"
            ")"
        )

    @property
    def model_fpath(self) -> pathlib.Path:
        return (
            self.dir_path
            / f"review_{self.review_id:06}"
            / self._model_fname.format(review_id=self.review_id)
        )

    def save(self) -> None:
        self.model_fpath.parent.mkdir(parents=True, exist_ok=True)
        with self.model_fpath.open(mode="wb") as f:
            joblib.dump(self.model, f)
        LOGGER.info(
            "<Review(id=%s)>: study ranking model saved to %s",
            self.review_id,
            self.model_fpath,
        )

    def learn_one(self, record: dict[str, str]) -> None:
        x = record[self.feature_col]
        y = record[self.target_col]
        self.model.learn_one(x, y)

    def learn_many(self, records: Iterable[dict[str, str]]) -> None:
        # HACK: this shit is broken in river v0.21!
        # but if we use custom ColandrTFIDF ...
        df = pd.DataFrame(data=records)
        X = df[self.feature_col].astype("string")
        y = df[self.target_col]
        self.model.learn_many(X, y)
        # for record in records:
        #     self.learn_one(record)

    def predict_one(
        self, record: dict[str, str], *, proba: bool = False
    ) -> bool | dict[bool, float]:
        x = record[self.feature_col]
        if not proba:
            return self.model.predict_one(x)
        else:
            return self.model.predict_proba_one(x)

    def predict_many(
        self, records: Iterable[dict[str, str]], *, proba: bool = False
    ) -> pd.Series:
        X = pd.DataFrame(data=records)[self.feature_col].astype("string")
        if not proba:
            return self.model.predict_many(X)
        else:
            return self.model.predict_proba_many(X)


@functools.lru_cache(maxsize=50)
def _load_study_ranker_model(
    review_id: int, fpath: str | pathlib.Path
) -> river.compose.Pipeline:
    try:
        with open(fpath, mode="rb") as f:
            _model = joblib.load(f)
    except IOError:
        LOGGER.info("<Review(id=%s)>: new study ranker model loaded ...", review_id)
        _model = _MODEL.clone()
    return _model


_MODEL = river.compose.Pipeline(
    (
        "featurizer",
        # river.feature_extraction.TFIDF(
        #     normalize=True, strip_accents=False, ngram_range=(1, 1)
        # ),
        ColandrTFIDF(normalize=True, strip_accents=False, ngram_range=(1, 1)),
    ),
    (
        "classifier",
        river.linear_model.LogisticRegression(
            optimizer=river.optim.SGD(lr=0.5),
            loss=river.optim.losses.BinaryFocalLoss(),
            initializer=river.optim.initializers.Zeros(),
            l2=0.001,
        ),
    ),
)
