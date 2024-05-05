import datetime
import itertools
import logging
import typing as t

import sqlalchemy as sa
import sqlalchemy.orm as sa_orm
import werkzeug.security
from sqlalchemy import event as sa_event
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import DynamicMapped as DM
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.orm import Mapped as M
from sqlalchemy.orm import WriteOnlyMapped as WOM
from sqlalchemy.orm import mapped_column as mapcol

from . import tasks, utils
from .extensions import db


# TODO: update relationship.lazy strategies
# https://docs.sqlalchemy.org/en/20/changelog/whatsnew_20.html#new-write-only-relationship-strategy-supersedes-dynamic

LOGGER = logging.getLogger(__name__)


class User(db.Model):
    __tablename__ = "users"

    # columns
    id: M[int] = mapcol(sa.Integer, primary_key=True, autoincrement=True)
    created_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    updated_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        onupdate=sa.func.now(),
        server_default=sa.func.now(),
        server_onupdate=sa.FetchedValue(),
    )
    name: M[str] = mapcol(sa.String(length=200))
    email: M[str] = mapcol(sa.String(length=200), unique=True, index=True)
    _password: M[str] = mapcol("password", sa.String(length=256))
    is_confirmed: M[bool] = mapcol(sa.Boolean, server_default=sa.false())
    is_admin: M[bool] = mapcol(sa.Boolean, server_default=sa.false())

    # relationships
    review_user_assoc: DM["ReviewUserAssoc"] = sa_orm.relationship(
        "ReviewUserAssoc",
        back_populates="user",
        cascade="all, delete",
        lazy="dynamic",
        order_by="ReviewUserAssoc.review_id",
    )
    reviews = association_proxy("review_user_assoc", "review")

    imports: WOM["Import"] = sa_orm.relationship(
        "Import", back_populates="user", lazy="write_only", passive_deletes=True
    )
    studies: WOM["Study"] = sa_orm.relationship(
        "Study", back_populates="user", lazy="write_only", passive_deletes=True
    )
    screenings: WOM["Screening"] = sa_orm.relationship(
        "Screening", back_populates="user", lazy="write_only", passive_deletes=True
    )

    def __repr__(self):
        return f"<User(id={self.id})>"

    @property
    def owned_reviews(self) -> list["Review"]:
        return [
            rua.review
            for rua in db.session.execute(
                sa.select(ReviewUserAssoc).filter_by(user_id=self.id, user_role="owner")
            ).scalars()
        ]

    @property
    def password(self) -> str:
        """User's (automatically hashed) password."""
        return self._password

    @password.setter
    def password(self, value):
        """Hash and set user password."""
        self._password = self.hash_password(value)

    def check_password(self, password: str) -> bool:
        return werkzeug.security.check_password_hash(self._password, password)

    @staticmethod
    def hash_password(password: str) -> str:
        return werkzeug.security.generate_password_hash(password, method="pbkdf2")


class Review(db.Model):
    __tablename__ = "reviews"

    # columns
    id: M[int] = mapcol(sa.Integer, primary_key=True, autoincrement=True)
    created_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    updated_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        onupdate=sa.func.now(),
        server_default=sa.func.now(),
        server_onupdate=sa.FetchedValue(),
    )
    name: M[str] = mapcol(sa.String(length=500))
    description: M[t.Optional[str]] = mapcol(sa.Text)
    status: M[str] = mapcol(sa.String(length=25), server_default="active")
    num_citation_screening_reviewers: M[int] = mapcol(
        sa.SmallInteger, server_default="1"
    )
    num_fulltext_screening_reviewers: M[int] = mapcol(
        sa.SmallInteger, server_default="1"
    )
    num_citations_included: M[int] = mapcol(sa.Integer, server_default="0")
    num_citations_excluded: M[int] = mapcol(sa.Integer, server_default="0")
    num_fulltexts_included: M[int] = mapcol(sa.Integer, server_default="0")
    num_fulltexts_excluded: M[int] = mapcol(sa.Integer, server_default="0")

    # relationships
    review_user_assoc = sa_orm.relationship(
        "ReviewUserAssoc",
        back_populates="review",
        cascade="all, delete",
        lazy="dynamic",
        order_by="ReviewUserAssoc.user_id",
    )
    users = association_proxy("review_user_assoc", "user")

    review_plan: M["ReviewPlan"] = sa_orm.relationship(
        "ReviewPlan", back_populates="review", lazy="select", passive_deletes=True
    )
    imports: WOM["Import"] = sa_orm.relationship(
        "Import", back_populates="review", lazy="write_only", passive_deletes=True
    )
    studies: WOM["Study"] = sa_orm.relationship(
        "Study", back_populates="review", lazy="write_only", passive_deletes=True
    )
    screenings: WOM["Screening"] = sa_orm.relationship(
        "Screening", back_populates="review", lazy="write_only", passive_deletes=True
    )
    dedupes: WOM["Dedupe"] = sa_orm.relationship(
        "Dedupe", back_populates="review", lazy="write_only", passive_deletes=True
    )
    data_extractions: WOM["DataExtraction"] = sa_orm.relationship(
        "DataExtraction",
        back_populates="review",
        lazy="write_only",
        passive_deletes=True,
    )

    def __init__(self, name, description=None):
        self.name = name
        self.description = description

    def __repr__(self):
        return f"<Review(id={self.id})>"

    @property
    def owners(self) -> list[User]:
        return [
            rua.user
            for rua in db.session.execute(
                sa.select(ReviewUserAssoc).filter_by(
                    review_id=self.id, user_role="owner"
                )
            ).scalars()
        ]


class ReviewUserAssoc(db.Model):
    __tablename__ = "review_user_assoc"

    review_id: M[int] = mapcol(
        sa.Integer, sa.ForeignKey("reviews.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: M[int] = mapcol(
        sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    user_role: M[t.Optional[str]] = mapcol(
        sa.Text, nullable=False, server_default=sa.text("'member'")
    )
    created_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    updated_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        onupdate=sa.func.now(),
        server_default=sa.func.now(),
        server_onupdate=sa.FetchedValue(),
    )

    review: M["Review"] = sa_orm.relationship(
        "Review", back_populates="review_user_assoc"
    )
    user: M["User"] = sa_orm.relationship("User", back_populates="review_user_assoc")

    def __init__(self, review: Review, user: User, user_role: t.Optional[str] = None):
        self.review = review
        self.user = user
        self.user_role = user_role

    def __repr__(self):
        return f"<ReviewUserAssoc(review_id={self.review_id}, user_id={self.user_id})>"


class ReviewPlan(db.Model):
    __tablename__ = "review_plans"

    # columns
    id: M[int] = mapcol(
        sa.BigInteger, sa.ForeignKey("reviews.id", ondelete="CASCADE"), primary_key=True
    )
    created_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    updated_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        onupdate=sa.func.now(),
        server_default=sa.func.now(),
        server_onupdate=sa.FetchedValue(),
    )
    objective: M[t.Optional[str]] = mapcol(sa.Text)
    research_questions = mapcol(
        postgresql.ARRAY(sa.String(length=300)),
        server_default="{}",
    )
    pico: M[dict[str, t.Any]] = mapcol(
        postgresql.JSONB(none_as_null=True), server_default="{}"
    )
    keyterms: M[list[dict[str, t.Any]]] = mapcol(
        postgresql.JSONB(none_as_null=True), server_default="{}"
    )
    selection_criteria: M[list[dict[str, t.Any]]] = mapcol(
        postgresql.JSONB(none_as_null=True), server_default="{}"
    )
    data_extraction_form: M[list[dict[str, t.Any]]] = mapcol(
        postgresql.JSONB(none_as_null=True), server_default="{}"
    )
    suggested_keyterms: M[dict[str, t.Any]] = mapcol(
        postgresql.JSONB(none_as_null=True), server_default="{}"
    )

    @hybrid_property
    def boolean_search_query(self):
        if not self.keyterms:
            return ""
        else:
            return utils.get_boolean_search_query(self.keyterms)

    # relationships
    review: M["Review"] = sa_orm.relationship(
        "Review", foreign_keys=[id], back_populates="review_plan", lazy="select"
    )

    def __init__(
        self,
        id_,
        objective=None,
        research_questions=None,
        pico=None,
        keyterms=None,
        selection_criteria=None,
        data_extraction_form=None,
    ):
        self.id = id_
        self.objective = objective
        self.research_questions = research_questions
        self.pico = pico
        self.keyterms = keyterms
        self.selection_criteria = selection_criteria
        self.data_extraction_form = data_extraction_form

    def __repr__(self):
        return f"<ReviewPlan(review_id={self.id})>"


class DataSource(db.Model):
    __tablename__ = "data_sources"
    __table_args__ = (
        db.UniqueConstraint(
            "source_type",
            "source_name",
            "source_url",
            name="uq_source_type_name_url",
            postgresql_nulls_not_distinct=True,
        ),
    )

    # columns
    id: M[int] = mapcol(sa.BigInteger, primary_key=True, autoincrement=True)
    created_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    source_type: M[str] = mapcol(sa.String(length=20), index=True)
    source_name: M[t.Optional[str]] = mapcol(sa.String(length=100), index=True)
    source_url: M[t.Optional[str]] = mapcol(sa.String(length=500))

    @hybrid_property
    def source_type_and_name(self):
        if self.source_name:
            return f"{self.source_type}: {self.source_name}"
        else:
            return self.source_type

    # relationships
    imports: DM["Import"] = sa_orm.relationship(
        "Import", back_populates="data_source", lazy="dynamic", passive_deletes=True
    )
    studies: DM["Study"] = sa_orm.relationship(
        "Study", back_populates="data_source", lazy="dynamic", passive_deletes=True
    )

    def __init__(self, source_type, source_name=None, source_url=None):
        self.source_type = source_type
        self.source_name = source_name
        self.source_url = source_url

    def __repr__(self):
        return f"<DataSource(id={self.id})>"


class Import(db.Model):
    __tablename__ = "imports"

    # columns
    id: M[int] = mapcol(sa.Integer, primary_key=True, autoincrement=True)
    created_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    review_id: M[int] = mapcol(
        sa.Integer, sa.ForeignKey("reviews.id", ondelete="CASCADE"), index=True
    )
    user_id: M[t.Optional[int]] = mapcol(
        sa.Integer, sa.ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    data_source_id: M[t.Optional[int]] = mapcol(
        sa.BigInteger, sa.ForeignKey("data_sources.id", ondelete="SET NULL")
    )
    record_type: M[str] = mapcol(sa.String(length=10))
    num_records: M[int] = mapcol(sa.Integer)
    status: M[t.Optional[str]] = mapcol(
        sa.String(length=20), server_default="not_screened"
    )

    # relationships
    review: M["Review"] = sa_orm.relationship(
        "Review", foreign_keys=[review_id], back_populates="imports", lazy="select"
    )
    user: M["User"] = sa_orm.relationship(
        "User", foreign_keys=[user_id], back_populates="imports", lazy="select"
    )
    data_source: M["DataSource"] = sa_orm.relationship(
        "DataSource",
        foreign_keys=[data_source_id],
        back_populates="imports",
        lazy="select",
    )

    def __init__(
        self, review_id, user_id, data_source_id, record_type, num_records, status=None
    ):
        self.review_id = review_id
        self.user_id = user_id
        self.data_source_id = data_source_id
        self.record_type = record_type
        self.num_records = num_records
        self.status = status

    def __repr__(self):
        return f"<Import(id={self.id})>"


class Study(db.Model):
    __tablename__ = "studies"

    # columns
    id: M[int] = mapcol(sa.BigInteger, primary_key=True, autoincrement=True)
    created_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    updated_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        onupdate=sa.func.now(),
        server_default=sa.func.now(),
        server_onupdate=sa.FetchedValue(),
    )
    citation: M[dict[str, t.Any]] = mapcol(
        postgresql.JSONB(none_as_null=True),
        nullable=True,  # TODO: False?
    )
    citation_text_content_vector_rep = mapcol(
        postgresql.ARRAY(sa.Float), server_default="{}"
    )
    fulltext: M[dict[str, t.Any]] = mapcol(
        postgresql.JSONB(none_as_null=True), nullable=True
    )
    user_id: M[t.Optional[int]] = mapcol(
        sa.Integer, sa.ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    review_id: M[int] = mapcol(
        sa.Integer, sa.ForeignKey("reviews.id", ondelete="CASCADE"), index=True
    )
    tags = mapcol(
        postgresql.ARRAY(sa.String(length=64)), server_default="{}", index=False
    )
    data_source_id: M[t.Optional[int]] = mapcol(
        sa.Integer, sa.ForeignKey("data_sources.id", ondelete="SET NULL"), index=True
    )
    dedupe_status: M[t.Optional[str]] = mapcol(
        sa.String(length=20), server_default="not_duplicate", index=True
    )
    citation_status: M[str] = mapcol(
        sa.String(length=20), server_default="not_screened", index=True
    )
    fulltext_status: M[str] = mapcol(
        sa.String(length=20), server_default="not_screened", index=True
    )
    data_extraction_status: M[str] = mapcol(
        sa.String(length=20), server_default="not_started", index=True
    )

    # relationships
    user: M["User"] = sa_orm.relationship(
        "User", foreign_keys=[user_id], back_populates="studies", lazy="select"
    )
    review: M["Review"] = sa_orm.relationship(
        "Review", foreign_keys=[review_id], back_populates="studies", lazy="select"
    )
    data_source: M["DataSource"] = sa_orm.relationship(
        "DataSource",
        foreign_keys=[data_source_id],
        back_populates="studies",
        lazy="select",
    )
    screenings: DM["Screening"] = sa_orm.relationship(
        "Screening",
        back_populates="study",
        lazy="dynamic",
        passive_deletes=True,
    )
    dedupe: M["Dedupe"] = sa_orm.relationship(
        "Dedupe", back_populates="study", lazy="joined", passive_deletes=True
    )
    data_extraction: M["DataExtraction"] = sa_orm.relationship(
        "DataExtraction", back_populates="study", lazy="joined", passive_deletes=True
    )

    def __repr__(self):
        return f"<Study(id={self.id})>"

    @hybrid_property
    def citation_text_content(self):
        return "\n\n".join(
            (
                self.citation.get("title", ""),
                self.citation.get("abstract", ""),
                ", ".join(self.citation.get("keywords", [])),
            )
        ).strip()

    @citation_text_content.expression
    def citation_text_content(cls):
        return db.func.concat_ws(
            "\n\n",
            cls.citation["title"],
            cls.citation["abstract"],
            db.func.array_to_string(cls.citation["keywords"], ", "),
        )

    @hybrid_property
    def citation_exclude_reasons(self):
        return self._exclude_reasons("citation")

    @hybrid_property
    def fulltext_exclude_reasons(self):
        return self._exclude_reasons("fulltext")

    def _exclude_reasons(self, stage: str):
        return sorted(
            set(
                itertools.chain.from_iterable(
                    screening.exclude_reasons or []
                    for screening in self.screenings
                    if screening.stage == stage
                )
            )
        )


class Screening(db.Model):
    __tablename__ = "screenings"
    __table_args__ = (
        db.UniqueConstraint(
            "user_id",
            "review_id",
            "study_id",
            "stage",
            name="uq_screenings_user_review_study_stage",
        ),
        db.Index("ix_screenings_study_id_stage", "study_id", "stage"),
    )

    # columns
    id: M[int] = mapcol(sa.BigInteger, primary_key=True, autoincrement=True)
    created_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    updated_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        onupdate=sa.func.now(),
        server_default=sa.func.now(),
        server_onupdate=sa.FetchedValue(),
    )
    user_id: M[t.Optional[int]] = mapcol(
        sa.Integer,
        sa.ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
    )
    review_id: M[int] = mapcol(
        sa.Integer,
        sa.ForeignKey("reviews.id", ondelete="CASCADE"),
        index=True,
    )
    study_id: M[int] = mapcol(
        sa.BigInteger,
        sa.ForeignKey("studies.id", ondelete="CASCADE"),
    )
    stage: M[str] = mapcol(sa.String(length=16))
    status: M[str] = mapcol(sa.String(length=20), index=True)
    exclude_reasons: M[list[str]] = mapcol(
        postgresql.ARRAY(sa.String(length=64)), nullable=True
    )

    # relationships
    user: M["User"] = sa_orm.relationship(
        "User",
        foreign_keys=[user_id],
        back_populates="screenings",
        lazy="select",
    )
    review: M["Review"] = sa_orm.relationship(
        "Review",
        foreign_keys=[review_id],
        back_populates="screenings",
        lazy="select",
    )
    study: M["Study"] = sa_orm.relationship(
        "Study",
        foreign_keys=[study_id],
        back_populates="screenings",
        lazy="select",
    )

    def __init__(
        self, user_id, review_id, study_id, stage, status, exclude_reasons=None
    ):
        self.user_id = user_id
        self.review_id = review_id
        self.study_id = study_id
        self.stage = stage
        self.status = status
        self.exclude_reasons = exclude_reasons

    def __repr__(self):
        return f"<Screening(study_id={self.study_id}, stage={self.stage})>"


class Dedupe(db.Model):
    __tablename__ = "dedupes"

    # columns
    id: M[int] = mapcol(sa.BigInteger, primary_key=True, autoincrement=True)
    created_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    study_id: M[int] = mapcol(
        sa.BigInteger,
        sa.ForeignKey("studies.id", ondelete="CASCADE"),
        index=True,
        unique=True,
    )
    review_id: M[int] = mapcol(
        sa.Integer, sa.ForeignKey("reviews.id", ondelete="CASCADE"), index=True
    )
    duplicate_of: M[t.Optional[int]] = mapcol(
        sa.BigInteger,  # sa.ForeignKey('studies.id', ondelete='SET NULL'),
    )
    duplicate_score: M[t.Optional[float]] = mapcol(sa.Float)

    # relationships
    study: M["Study"] = sa_orm.relationship(
        "Study", foreign_keys=[study_id], back_populates="dedupe", lazy="select"
    )
    review: M["Review"] = sa_orm.relationship(
        "Review", foreign_keys=[review_id], back_populates="dedupes", lazy="select"
    )

    def __init__(self, study_id, review_id, duplicate_of, duplicate_score):
        self.study_id = study_id
        self.review_id = review_id
        self.duplicate_of = duplicate_of
        self.duplicate_score = duplicate_score

    def __repr__(self):
        return f"<Dedupe(study_id={self.study_id})>"


class DataExtraction(db.Model):
    __tablename__ = "data_extractions"

    # columns
    id: M[int] = mapcol(sa.BigInteger, primary_key=True, autoincrement=True)
    created_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    updated_at: M[datetime.datetime] = mapcol(
        sa.DateTime(timezone=True),
        onupdate=sa.func.now(),
        server_default=sa.func.now(),
        server_onupdate=sa.FetchedValue(),
    )
    study_id: M[int] = mapcol(
        sa.BigInteger,
        sa.ForeignKey("studies.id", ondelete="CASCADE"),
        index=True,
        unique=True,
    )
    review_id: M[int] = mapcol(
        sa.Integer,
        sa.ForeignKey("reviews.id", ondelete="CASCADE"),
        index=True,
    )
    extracted_items: M[t.Optional[list[dict[str, t.Any]]]] = mapcol(
        postgresql.JSONB(none_as_null=True), server_default="{}"
    )

    # relationships
    study: M["Study"] = sa_orm.relationship(
        "Study",
        foreign_keys=[study_id],
        back_populates="data_extraction",
        lazy="select",
    )
    review: M["Review"] = sa_orm.relationship(
        "Review",
        foreign_keys=[review_id],
        back_populates="data_extractions",
        lazy="select",
    )

    def __init__(self, study_id, review_id, extracted_items=None):
        self.study_id = study_id
        self.review_id = review_id
        self.extracted_items = extracted_items

    def __repr__(self):
        return f"<DataExtraction(study_id={self.study_id})>"


# EVENTS

# NOTE: apparently this does not work in sqlalchemy v2 :/
# @sa_event.listens_for(db.Model, "after_update")
# def update_updated_at(mapper, connection, target):
#     updated_at = connection.execute(sa.select(sa.func.now())).scalar()
#     LOGGER.warning("%s.updated_at = %s", target, updated_at)
#     if hasattr(target, "updated_at"):
#         target.updated_at = updated_at


@sa_event.listens_for(Review, "after_insert")
def insert_review_plan(mapper, connection, target):
    review_plan = ReviewPlan(target.id)
    connection.execute(sa.insert(ReviewPlan).values(id=target.id))
    LOGGER.info("inserted %s and %s", target, review_plan)


@sa_event.listens_for(Screening, "after_insert")
@sa_event.listens_for(Screening, "after_delete")
@sa_event.listens_for(Screening, "after_update")
def update_study_status(mapper, connection, target):
    review_id = target.review_id
    study_id = target.study_id
    study = target.study
    # TODO(burton): you added this so that conftest populate_db func would work
    # for reasons unknown, the target here didn't have a loaded citation object
    # but this is _probably_ a bad thing, and you should find a way to fix it
    if study is None:
        study = db.session.execute(
            sa.select(Study).filter_by(id=study_id)
        ).scalar_one_or_none()
    assert isinstance(study, Study)  # type guard
    # prep stage-specific variables
    stage = target.stage
    if stage == "citation":
        num_reviewers = study.review.num_citation_screening_reviewers
        study_status_col = Study.citation_status
        study_status_col_str = "citation_status"
        review_num_included_col = Review.num_citations_included
        review_num_included_col_str = "num_citations_included"
    else:
        num_reviewers = study.review.num_fulltext_screening_reviewers
        study_status_col = Study.fulltext_status
        study_status_col_str = "fulltext_status"
        review_num_included_col = Review.num_fulltexts_included
        review_num_included_col_str = "num_fulltexts_included"
    # get the current (soon to be *old*) status of the study
    old_status = connection.execute(
        sa.select(study_status_col).where(Study.id == study_id)
    ).fetchone()[0]
    # compute the new status, and update the study accordingly
    status = utils.assign_status(
        [
            screening.status
            for screening in db.session.execute(
                sa.select(Screening).filter_by(study_id=study_id, stage=stage)
            ).scalars()
        ],
        num_reviewers,
    )
    connection.execute(
        sa.update(Study)
        .where(Study.id == study_id)
        .values({study_status_col_str: status})
    )
    LOGGER.info("%s => %s with %s status = %s", target, study, stage, status)

    # we may have to update our counts for review num_*_included / num_*_excluded
    _update_review_num_counts(
        old_status=old_status,
        status=status,
        review_id=review_id,
        review_num_included_col=review_num_included_col,
        review_num_included_col_str=review_num_included_col_str,
        connection=connection,
    )

    if stage == "citation":
        # get rid of any contrary fulltext screenings
        if status != "included":
            connection.execute(
                sa.delete(Screening)
                .where(Screening.study_id == study_id)
                .where(Screening.stage == "fulltext")
            )
            LOGGER.info(
                "deleted all <Screening(study_id=%s, stage='fulltext')>", study_id
            )
            # TODO: do we also need to update review.num_fulltexts_included/excluded?
        # update review models, as needed
        status_counts = connection.execute(
            sa.select(
                Review.num_citations_included, Review.num_citations_excluded
            ).where(Review.id == review_id)
        ).fetchone()
        LOGGER.info(
            "<Review(id=%s)> citation_status counts = %s", review_id, status_counts
        )
        n_included, n_excluded = status_counts
        # if at least 25 citations have been included AND excluded
        # and only once every 25 included citations
        # (re-)compute the suggested keyterms
        if n_included >= 25 and n_excluded >= 25 and n_included % 25 == 0:
            sample_size = min(n_included, n_excluded)
            tasks.suggest_keyterms.apply_async(args=[review_id, sample_size])
        # if at least 100 citations have been included AND excluded
        # and only once every 50 included citations
        # (re-)train a citation ranking model
        if n_included >= 100 and n_excluded >= 100 and n_included % 50 == 0:
            tasks.train_citation_ranking_model.apply_async(args=[review_id])
    elif stage == "fulltext":
        # we may have to insert or delete a corresponding data extraction record
        data_extraction = connection.execute(
            sa.select(DataExtraction).where(DataExtraction.study_id == study_id)
        ).first()
        # data_extraction_inserted_or_deleted = False
        if status == "included" and data_extraction is None:
            connection.execute(
                sa.insert(DataExtraction).values(study_id=study_id, review_id=review_id)
            )
            LOGGER.info("inserted <DataExtraction(study_id=%s)>", study_id)
            # data_extraction_inserted_or_deleted = True
        elif status != "included" and data_extraction is not None:
            connection.execute(
                sa.delete(DataExtraction).where(DataExtraction.study_id == study_id)
            )
            LOGGER.info("deleted <DataExtraction(study_id=%s)>", study_id)
            # data_extraction_inserted_or_deleted = True
        # TODO: should we do something now?
        # if data_extraction_inserted_or_deleted is True:
        #     with connection.begin():
        #         status_counts = connection.execute(
        #             sa.select(Review.num_fulltexts_included, Review.num_fulltexts_excluded)\
        #             .where(Review.id == review_id)
        #             ).fetchone()
        #         LOGGER.info(
        #             '<Review(id=%s)> fulltext_status counts = %s',
        #             review_id, status_counts)
        #         n_included, n_excluded = status_counts


def _update_review_num_counts(
    *,
    old_status: str,
    status: str,
    review_id: int,
    review_num_included_col: InstrumentedAttribute[int],
    review_num_included_col_str: str,
    connection,
):
    if old_status != status:
        if old_status == "included":  # decrement
            connection.execute(
                sa.update(Review)
                .where(Review.id == review_id)
                .values({review_num_included_col_str: review_num_included_col - 1})
            )
        elif status == "included":  # increment
            connection.execute(
                sa.update(Review)
                .where(Review.id == review_id)
                .values({review_num_included_col_str: review_num_included_col + 1})
            )
        elif old_status == "excluded":  # decrement
            connection.execute(
                sa.update(Review)
                .where(Review.id == review_id)
                .values({review_num_included_col_str: review_num_included_col - 1})
            )
        elif status == "excluded":  # increment
            connection.execute(
                sa.update(Review)
                .where(Review.id == review_id)
                .values({review_num_included_col_str: review_num_included_col + 1})
            )
