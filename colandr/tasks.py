import itertools
import os
import typing as t

import redis
import redis.client
import redis.lock
import sqlalchemy as sa
from celery import current_app as current_celery_app
from celery import shared_task
from celery.utils.log import get_task_logger
from flask import current_app
from flask_mail import Message

from . import models
from .apis.schemas import ReviewPlanSuggestedKeyterms
from .extensions import db, mail
from .lib.models import Deduper, Ranker, StudyRanker
from .lib.nlp import hack
from .lib.nlp import utils as nlp_utils


LOGGER = get_task_logger(__name__)


def _get_redis_lock(lock_id: str, timeout: int = 120) -> redis.lock.Lock:
    redis_conn = _get_redis_conn()
    return redis_conn.lock(lock_id, timeout=timeout, sleep=1.0, blocking=True)


def _get_redis_conn() -> redis.client.Redis:
    redis_conn = current_celery_app.backend.client  # type: ignore
    assert isinstance(redis_conn, redis.client.Redis)  # type guard
    return redis_conn


@shared_task
def send_email(recipients, subject, text_body, html_body):
    msg = Message(
        subject=f"{current_app.config['MAIL_SUBJECT_PREFIX']} {subject}",
        sender=current_app.config["MAIL_DEFAULT_SENDER"],
        recipients=recipients,
        body=text_body,
        html=html_body,
    )
    mail.send(msg)


@shared_task
def remove_unconfirmed_user(email: str):
    user = db.session.execute(
        sa.select(models.User).filter_by(email=email)
    ).scalar_one_or_none()
    if user and user.is_confirmed is False:
        db.session.delete(user)
        db.session.commit()


@shared_task
def deduplicate_citations(review_id: int):
    lock = _get_redis_lock(f"deduplicate_ciations__review-{review_id}")
    lock.acquire()

    stmt = sa.select(sa.func.max(models.Study.created_at)).where(
        models.Study.review_id == review_id
    )
    max_created_at = db.session.execute(stmt).scalar()
    # no citations? cancel dedupe
    if max_created_at is None:
        LOGGER.warning(
            "<Review(id=%s)>: no citations found; skipping dedupe ...", review_id
        )
        lock.release()
        return

    stmt = sa.select(sa.func.max(models.Dedupe.created_at)).where(
        models.Dedupe.review_id == review_id
    )
    most_recent_dedupe = db.session.execute(stmt).scalar()
    # no citations added since most recent dedupe? cancel dedupe
    if most_recent_dedupe and most_recent_dedupe > max_created_at:
        LOGGER.info(
            "<Review(id=%s)>: all citations already deduped; skipping dedupe ...",
            review_id,
        )
        lock.release()
        return

    deduper = Deduper.load(
        current_app.config["DEDUPE_MODELS_DIR"], num_cores=1, in_memory=False
    )

    # remove dedupe rows for this review
    # which we'll add back with the latest citations included
    stmt = sa.delete(models.Dedupe).where(models.Dedupe.review_id == review_id)
    result = db.session.execute(stmt)
    rows_deleted = result.rowcount
    LOGGER.debug(
        "<Review(id=%s)>: deleted %s rows from %s",
        review_id,
        rows_deleted,
        models.Dedupe.__tablename__,
    )

    stmt = sa.select(
        models.Study.id,
        models.Study.citation["type_of_reference"].label("type_of_reference"),
        models.Study.citation["title"].label("title"),
        models.Study.citation["pub_year"].label("pub_year"),
        models.Study.citation["authors"].label("authors"),
        models.Study.citation["abstract"].label("abstract"),
        models.Study.citation["doi"].label("doi"),
    ).where(models.Study.review_id == review_id)
    # results = db.session.execute(stmt).mappings() instead ?
    results = (row._asdict() for row in db.session.execute(stmt))
    preproc_data = deduper.preprocess_data(results, id_key="id")

    # TODO: decide on suitable value for threshold; higher => higher precision
    clustered_dupes = deduper.model.partition(preproc_data, threshold=0.5)
    try:
        LOGGER.info(
            "<Review(id=%s)>: found %s duplicate clusters",
            review_id,
            len(clustered_dupes),  # type: ignore
        )
    # TODO: figure out if this is ever a generator instead
    except TypeError:
        LOGGER.info("<Review(id=%s)>: found duplicate clusters", review_id)

    # get *all* citation ids for this review, as well as included/excluded
    stmt = sa.select(models.Study.id).where(models.Study.review_id == review_id)
    all_sids = set(db.session.execute(stmt).scalars().all())
    stmt = (
        sa.select(models.Study.id)
        .where(models.Study.review_id == review_id)
        # .where(models.Study.citation_status.in_(["included", "excluded"]))
        .where(models.Study.citation_status == sa.any_(["included", "excluded"]))
    )
    incl_excl_sids = set(db.session.execute(stmt).scalars().all())

    duplicate_sids = set()
    studies_to_update = []
    dedupes_to_insert = []
    for sids, scores in clustered_dupes:
        int_sids = [int(sid) for sid in sids]  # convert from numpy.int64
        sid_scores = {sid: float(score) for sid, score in zip(int_sids, scores)}
        # already an in/excluded citation in this dupe cluster?
        # take the first one to be "canonical"
        if any(sid in incl_excl_sids for sid in int_sids):
            canonical_study_id = sorted(set(int_sids).intersection(incl_excl_sids))[0]
        # otherwise, take the "most complete" citation in the cluster as "canonical"
        else:
            stmt = (
                sa.select(
                    models.Study.id,
                    (
                        sa.case((models.Study.citation["title"] == None, 1))
                        + sa.case((models.Study.citation["abstract"] == None, 1))
                        + sa.case((models.Study.citation["pub_year"] == None, 1))
                        + sa.case((models.Study.citation["pub_month"] == None, 1))
                        + sa.case((models.Study.citation["authors"] == [], 1))
                        + sa.case((models.Study.citation["keywords"] == [], 1))
                        + sa.case(
                            (models.Study.citation["type_of_reference"] == None, 1)
                        )
                        + sa.case((models.Study.citation["journal_name"] == None, 1))
                        + sa.case((models.Study.citation["issue_number"] == None, 1))
                        + sa.case((models.Study.citation["doi"] == None, 1))
                        + sa.case((models.Study.citation["issn"] == None, 1))
                        + sa.case((models.Study.citation["publisher"] == None, 1))
                        + sa.case((models.Study.citation["language"] == None, 1))
                    ).label("n_null_cols"),
                )
                .where(models.Study.review_id == review_id)
                # .where(models.Study.id.in_(int_sids))
                .where(models.Study.id == sa.any_(int_sids))
                .order_by(sa.text("n_null_cols ASC"))
                .limit(1)
            )
            result = db.session.execute(stmt).first()
            assert result is not None
            canonical_study_id = result.id

        for sid, score in sid_scores.items():
            if sid != canonical_study_id:
                duplicate_sids.add(sid)
                studies_to_update.append({"id": sid, "dedupe_status": "duplicate"})
                dedupes_to_insert.append(
                    {
                        "study_id": sid,
                        "review_id": review_id,
                        "duplicate_of": canonical_study_id,
                        "duplicate_score": score,
                    }
                )
    non_duplicate_sids = all_sids - duplicate_sids
    studies_to_update.extend(
        {"id": sid, "dedupe_status": "not_duplicate"} for sid in non_duplicate_sids
    )

    db.session.execute(sa.update(models.Study), studies_to_update)
    db.session.execute(sa.insert(models.Dedupe), dedupes_to_insert)
    db.session.commit()
    LOGGER.info(
        "<Review(id=%s)>: found %s duplicate and %s non-duplicate citations",
        review_id,
        len(duplicate_sids),
        len(non_duplicate_sids),
    )

    lock.release()


@shared_task
def get_citations_text_content_vectors(review_id: int):
    lock = _get_redis_lock(f"get_citations_text_content_vectors__review-{review_id}")
    lock.acquire()

    stmt = (
        sa.select(models.Study.id, models.Study.citation_text_content)
        .where(models.Study.review_id == review_id)
        .where(models.Study.citation_text_content_vector_rep == [])
        .order_by(models.Study.id)
    )
    results = db.session.execute(stmt)
    ids, texts = zip(*results)
    docs = nlp_utils.process_texts_into_docs(
        texts,
        max_len=1000,
        min_prob=0.75,
        fallback_lang="en",
        exclude=("parser", "ner"),
    )
    cvs = (doc.vector.tolist() if doc is not None else None for doc in docs)
    citations_to_update = [
        {"id": id_, "text_content_vector_rep": cv}
        for id_, cv in zip(ids, cvs)
        if cv is not None
    ]
    if not citations_to_update:
        LOGGER.warning(
            "<Review(id=%s)>: no citation text_content_vector_reps to update",
            review_id,
        )
        lock.release()
        return

    db.session.execute(sa.update(models.Study), citations_to_update)
    db.session.commit()
    LOGGER.info(
        "<Review(id=%s)>: %s citation text_content_vector_reps updated",
        review_id,
        len(citations_to_update),
    )

    lock.release()


@shared_task
def get_fulltext_text_content_vector(fulltext_id: int):
    stmt = sa.select(models.Study.fulltext).where(models.Study.id == fulltext_id)
    fulltext = db.session.execute(stmt).scalar_one_or_none()
    if not fulltext or not fulltext.get("text_content"):
        LOGGER.warning(
            "no fulltext text content found for <Study(study_id=%s)>", fulltext_id
        )
        return

    docs = nlp_utils.process_texts_into_docs(
        [fulltext["text_content"]],
        max_len=3000,
        min_prob=0.75,
        fallback_lang=None,
        exclude=("parser", "ner"),
    )
    doc = next(iter(docs))
    text_content_vector_rep = doc.vector.tolist() if doc is not None else None
    if text_content_vector_rep is None:
        LOGGER.warning(
            "unable to get word vectors for <Study(study_id=%s)>", fulltext_id
        )
        return

    fulltext["text_content_vector_rep"] = text_content_vector_rep
    stmt = (
        sa.update(models.Study)
        .where(models.Study.id == fulltext_id)
        .values(fulltext=fulltext)
    )
    db.session.execute(stmt)
    db.session.commit()


@shared_task
def suggest_keyterms(review_id: int, sample_size: int):
    lock = _get_redis_lock(f"suggest_keyterms__review-{review_id}")
    lock.acquire()

    LOGGER.info(
        "<Review(id=%s)>: computing keyterms with sample size = %s",
        review_id,
        sample_size,
    )

    # get random sample of included citations
    stmt = (
        sa.select(models.Study.citation_status, models.Study.citation_text_content)
        .where(models.Study.review_id == review_id)
        .where(models.Study.citation_status == "included")
        .order_by(sa.func.random())
        .limit(sample_size)
    )
    included = db.session.execute(stmt).all()
    # get random sample of excluded citations
    stmt = (
        sa.select(models.Study.citation_status, models.Study.citation_text_content)
        .where(models.Study.review_id == review_id)
        .where(models.Study.citation_status == "excluded")
        .order_by(sa.func.random())
        .limit(sample_size)
    )
    excluded = db.session.execute(stmt).all()

    # munge the results into the form needed by textacy
    included_vec = [
        status == "included" for status, _ in itertools.chain(included, excluded)
    ]
    docs = nlp_utils.process_texts_into_docs(
        (text for _, text in itertools.chain(included, excluded)),
        max_len=3000,
        min_prob=0.75,
        fallback_lang=None,
        exclude=("parser", "ner"),
    )
    terms_lists = (
        doc._.to_terms_list(include_pos={"NOUN", "VERB"}, as_strings=True)
        for doc in docs
        if doc is not None
    )
    # run the analysis!
    incl_keyterms, excl_keyterms = hack.most_discriminating_terms(
        terms_lists, included_vec, top_n_terms=50
    )

    # munge results into form expected by the database, and validate
    suggested_keyterms = {
        "sample_size": sample_size,
        "incl_keyterms": incl_keyterms,
        "excl_keyterms": excl_keyterms,
    }
    errors = ReviewPlanSuggestedKeyterms().validate(suggested_keyterms)
    if errors:
        lock.release()
        raise Exception
    LOGGER.info(
        "<Review(id=%s)>: suggested keyterms: %s", review_id, suggested_keyterms
    )
    # update the review plan
    stmt = (
        sa.update(models.ReviewPlan)
        .where(models.ReviewPlan.id == review_id)
        .values(suggested_keyterms=suggested_keyterms)
    )
    db.session.execute(stmt)
    db.session.commit()

    lock.release()


@shared_task
def train_study_ranker_model(review_id: int, screening_id: t.Optional[int] = None):
    lock = _get_redis_lock(f"train_study_ranker_model__review-{review_id}")
    lock.acquire()

    study_ranker = StudyRanker(review_id, current_app.config["RANKER_MODELS_DIR"])
    if screening_id is None or not study_ranker.model_fpath.exists():
        _train_study_ranker_model_from_scratch(study_ranker, review_id)
    else:
        _train_study_ranker_model_from_screening(study_ranker, screening_id)

    lock.release()


def _train_study_ranker_model_from_scratch(study_ranker: StudyRanker, review_id: int):
    LOGGER.info("<Review(id=%s)>: training study ranker model from scratch", review_id)
    # get target+text for studies that have been fully screened at either stage
    # preferring fulltext- over citation-stage screening since it's based on more info
    stmt1 = sa.select(
        (
            sa.case(
                (
                    models.Study.fulltext_status.in_(["included", "excluded"]),
                    models.Study.fulltext_status,
                ),
                else_=models.Study.citation_status,
            )
            == "included"
        ).label("target"),
        sa.case(
            (
                models.Study.fulltext_status.in_(["included", "excluded"]),
                sa.func.substring(
                    models.Study.fulltext["text_content"].astext, 0, 5000
                ),
            ),
            else_=models.Study.citation_text_content,
        ).label("text"),
    ).where(
        models.Study.review_id == review_id,
        models.Study.dedupe_status == "not_duplicate",
        models.Study.citation_status.in_(["included", "excluded"]),
        models.Study.fulltext_status.in_(["included", "excluded", "not_screened"]),
    )
    # get target+text for studies that have been partially screened at either stage
    # leveraging study text corresponding to the screening's stage
    stmt2 = (
        sa.select(
            (models.Screening.status == "included").label("target"),
            sa.case(
                (
                    models.Screening.stage == "fulltext",
                    sa.func.substring(
                        models.Study.fulltext["text_content"].astext, 0, 5000
                    ),
                ),
                else_=models.Study.citation_text_content,
            ).label("text"),
        )
        .select_from(models.Study)
        .join(models.Screening, models.Study.id == models.Screening.study_id)
        .where(
            sa.case(
                (
                    models.Screening.stage == "fulltext",
                    ~models.Study.fulltext_status.in_(["included", "excluded"]),
                ),
                else_=~models.Study.citation_status.in_(["included", "excluded"]),
            )
        )
    )
    # union outputs from both cases
    stmt = stmt1.union_all(stmt2)
    records = (row._asdict() for row in db.session.execute(stmt))
    study_ranker.learn_many(records)
    study_ranker.save()


def _train_study_ranker_model_from_screening(
    study_ranker: StudyRanker, screening_id: int
):
    LOGGER.info("training study ranker model from <Screening(id=%s)>", screening_id)
    screening = db.session.get(models.Screening, screening_id)
    if screening is None:
        LOGGER.warning(
            "<Screening(id=%s)> not found; study ranker model training not possible",
            screening_id,
        )
        return

    study = screening.study
    target = screening.status == "included"
    text: str = (
        study.fulltext.get("text_content", "")
        if screening.stage == "fulltext"
        else study.citation_text_content
    )
    study_ranker.learn_one({"text": text, "target": target})
    # TODO: decide if we want to save model after every single screening
    # saving takes ~20x longer than learning, so it's not "cheap"
    # maybe we could get away with saving only every ~10 screenings
    study_ranker.save()
