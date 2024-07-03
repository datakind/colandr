import functools
import logging
import pathlib
import re
import typing as t
import urllib.parse
from collections.abc import Iterable

import dedupe
from dedupe import variables
from textacy import preprocessing

from .. import utils


LOGGER = logging.getLogger(__name__)

RE_DOI_HTTP = re.compile(r"^https?(://)?", flags=re.IGNORECASE)

SETTINGS_FNAME = "deduper_settings"
TRAINING_FNAME = "deduper_training.json"
VARIABLES: list[variables.base.Variable] = [
    variables.Exact("type_of_reference"),
    variables.String("doi", has_missing=True),
    variables.String("title", name="title"),
    variables.String("authors_joined", has_missing=True, name="authors_joined"),
    variables.Set("authors_initials", has_missing=True, name="authors_initials"),
    variables.Exact("pub_year", has_missing=True, name="pub_year"),
    variables.String("journal_name", has_missing=True, name="journal_name"),
    variables.Exact("journal_volume", has_missing=True, name="journal_volume"),
    variables.Exact(
        "journal_issue_number", has_missing=True, name="journal_issue_number"
    ),
    variables.String("issn", has_missing=True, name="issn"),
    variables.Text("abstract", has_missing=True),
    variables.Interaction("journal_name", "pub_year"),
    variables.Interaction("journal_name", "journal_volume", "journal_issue_number"),
    variables.Interaction("issn", "pub_year"),
    variables.Interaction("title", "authors_joined"),
]


class Deduper:
    def __init__(
        self,
        *,
        settings_fpath: t.Optional[str | pathlib.Path] = None,
        num_cores: int = 1,
        in_memory: bool = False,
    ):
        self.settings_fpath = settings_fpath
        self.num_cores = num_cores
        self.in_memory = in_memory

    @classmethod
    def load(
        cls, dir_path: str | pathlib.Path, num_cores: int = 1, in_memory: bool = False
    ):
        settings_fpath = utils.to_path(dir_path) / SETTINGS_FNAME
        return cls(
            settings_fpath=settings_fpath, num_cores=num_cores, in_memory=in_memory
        )

    @functools.cached_property
    def model(self) -> dedupe.Dedupe | dedupe.StaticDedupe:
        if self.settings_fpath is None:
            _model = dedupe.Dedupe(
                VARIABLES,  # type: ignore
                num_cores=self.num_cores,
                in_memory=self.in_memory,
            )
        else:
            with open(self.settings_fpath, mode="rb") as f:
                _model = dedupe.StaticDedupe(
                    f, num_cores=self.num_cores, in_memory=self.in_memory
                )
        return _model

    def preprocess_data(
        self,
        data: Iterable[dict[str, t.Any]],
        id_key: str,
    ) -> dict[t.Any, dict[str, t.Any]]:
        fields = [pv.field for pv in self.model.data_model.field_variables]
        LOGGER.info("preprocessing data with fields %s ...", fields)
        return {record.pop(id_key): self._preprocess_record(record) for record in data}

    def _preprocess_record(self, record: dict[str, t.Any]) -> dict[str, t.Any]:
        # base fields
        record = {
            "type_of_reference": (
                record["type_of_reference"].strip().lower()
                if record.get("type_of_reference")
                else None
            ),
            "doi": (_sanitize_doi(record["doi"]) if record.get("doi") else None),
            "title": (
                _standardize_str(record["title"]) if record.get("title") else None
            ),
            "authors": (
                tuple(
                    sorted(
                        _standardize_str(author.replace("-", " "))
                        for author in record["authors"]
                    )
                )
                if record.get("authors")
                else None
            ),
            "pub_year": record.get("pub_year"),
            "journal_name": (
                preprocessing.remove.brackets(
                    _standardize_str(record["journal_name"]), only="round"
                )
                if record.get("journal_name")
                else None
            ),
            "journal_volume": record.get("volume"),
            "journal_issue_number": record.get("issue_number"),
            "issn": record["issn"].strip().lower() if record.get("issn") else None,
            "abstract": (
                _standardize_str(record["abstract"][:500])  # truncated for performance
                if record.get("abstract")
                else None
            ),
        }
        # derivative fields
        if record.get("authors"):
            record["authors_initials"] = tuple(
                "".join(name[0] for name in author.split())
                for author in record["authors"]
            )
            record["authors_joined"] = " ".join(record["authors"])
        else:
            record["authors_initials"] = None
            record["authors_joined"] = None
        return record

    def fit(
        self,
        data: dict[t.Any, dict[str, t.Any]],
        training_fpath: t.Optional[str | pathlib.Path] = None,
        recall: float = 1.0,
        index_predicates: bool = True,
    ) -> "Deduper":
        if isinstance(self.model, dedupe.StaticDedupe):
            raise TypeError("deduper loaded from a settings file can't be re-fit")
        LOGGER.info("preparing model %s for training ...", self.model)
        if training_fpath is None:
            self.model.prepare_training(data)
        else:
            training_fpath = utils.to_path(training_fpath)
            with training_fpath.open(mode="r") as f:
                self.model.prepare_training(data, training_file=f)

        dedupe.console_label(self.model)
        LOGGER.info("training model on labeled examples ...")
        self.model.train(recall, index_predicates)
        return self

    def predict(
        self, data: dict[t.Any, dict[str, t.Any]], threshold: float = 0.5
    ) -> list[tuple[tuple, tuple[float, ...]]]:
        return self.model.partition(data, threshold=threshold)  # type: ignore

    def save(self, dir_path: str | pathlib.Path) -> None:
        if isinstance(self.model, dedupe.StaticDedupe):
            raise TypeError("deduper loaded from a settings file can't be re-saved")
        dir_path = utils.to_path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info("saving deduper model setings and training data to %s", dir_path)
        with (dir_path / SETTINGS_FNAME).open(mode="wb") as f:
            self.model.write_settings(f)
        with (dir_path / TRAINING_FNAME).open(mode="w") as f:
            self.model.write_training(f)


def _sanitize_doi(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("http://") or value.startswith("https://"):
        value = urllib.parse.unquote(value)
        value = RE_DOI_HTTP.sub("", value)
    return value


_standardize_str = preprocessing.make_pipeline(
    functools.partial(
        preprocessing.remove.punctuation, only=[".", "?", "!", ",", ";", "—"]
    ),
    preprocessing.normalize.quotation_marks,
    preprocessing.normalize.whitespace,
    str.lower,
)
