import random
from operator import itemgetter

import flask_jwt_extended as jwtext
import sqlalchemy as sa
from flask import current_app
from flask_restx import Namespace, Resource
from marshmallow import fields as ma_fields
from marshmallow.validate import Length, OneOf, Range
from sqlalchemy.sql import operators
from webargs.fields import DelimitedList
from webargs.flaskparser import use_args, use_kwargs

from ... import models
from ...extensions import db
from ...lib import constants
from ...lib.models import StudyRanker
from ...lib.nlp import reviewer_terms
from ..errors import forbidden_error, not_found_error
from ..schemas import StudySchema
from ..swagger import study_model


ns = Namespace("studies", path="/studies", description="get, delete, update studies")


@ns.route("/<int:id>")
@ns.doc(
    summary="get, delete, and modify data for single studies",
    produces=["application/json"],
)
class StudyResource(Resource):
    @ns.doc(
        params={
            "fields": {
                "in": "query",
                "type": "string",
                "description": "comma-delimited list-as-string of review fields to return",
            },
        },
        responses={
            200: "successfully got study record",
            403: "current app user forbidden to get study record",
            404: "no study with matching id was found",
        },
    )
    @use_kwargs(
        {
            "id": ma_fields.Int(
                required=True, validate=Range(min=1, max=constants.MAX_BIGINT)
            )
        },
        location="view_args",
    )
    @use_kwargs(
        {"fields": DelimitedList(ma_fields.String, delimiter=",", load_default=None)},
        location="query",
    )
    @jwtext.jwt_required()
    def get(self, id, fields):
        """get record for a single study by id"""
        current_user = jwtext.get_current_user()
        study = db.session.get(models.Study, id)
        if not study:
            return not_found_error(f"<Study(id={id})> not found")
        if not _is_allowed(current_user, study.review_id):
            return forbidden_error(f"{current_user} forbidden to get this study")
        if fields and "id" not in fields:
            fields.append("id")
        current_app.logger.debug("got %s", study)
        return StudySchema(only=fields).dump(study)

    @ns.doc(
        responses={
            204: "successfully deleted study record",
            403: "current app user forbidden to delete study record",
            404: "no study with matching id was found",
        },
    )
    @use_kwargs(
        {
            "id": ma_fields.Int(
                required=True, validate=Range(min=1, max=constants.MAX_BIGINT)
            ),
        },
        location="view_args",
    )
    @jwtext.jwt_required(fresh=True)
    def delete(self, id):
        """delete record for a single study by id"""
        current_user = jwtext.get_current_user()
        study = db.session.get(models.Study, id)
        if not study:
            return not_found_error(f"<Study(id={id})> not found")
        if not _is_allowed(current_user, study.review_id):
            return forbidden_error(f"{current_user} forbidden to delete this study")
        db.session.delete(study)
        db.session.commit()
        current_app.logger.info("deleted %s", study)
        return "", 204

    @ns.doc(
        expect=(study_model, "study data to be modified"),
        responses={
            200: "study data was modified",
            403: "current app user forbidden to modify study; specified field may not be modified",
            404: "no study with matching id was found",
        },
    )
    @use_args(StudySchema(only=["data_extraction_status", "tags"]), location="json")
    @use_kwargs(
        {
            "id": ma_fields.Int(
                required=True, validate=Range(min=1, max=constants.MAX_BIGINT)
            ),
        },
        location="view_args",
    )
    @jwtext.jwt_required()
    def put(self, args, id):
        """modify record for a single study by id"""
        current_user = jwtext.get_current_user()
        study = db.session.get(models.Study, id)
        if not study:
            return not_found_error(f"<Study(id={id})> not found")
        if not _is_allowed(current_user, study.review_id):
            return forbidden_error(f"{current_user} forbidden to modify this study")
        for key, value in args.items():
            if key == "data_extraction_status":
                if study.fulltext_status != "included":
                    return forbidden_error(
                        f"<Study(id={id})> data_extraction_status can't be set "
                        "until fulltext has passed screening"
                    )
            setattr(study, key, value)
        db.session.commit()
        current_app.logger.info("modified %s", study)
        return StudySchema().dump(study)


@ns.route("")
@ns.doc(
    summary="get collections of matching studies",
    produces=["application/json"],
)
class StudiesResource(Resource):
    @ns.doc(
        params={
            "review_id": {
                "in": "query",
                "type": "integer",
                "required": True,
                "description": "unique identifier for review whose studies are to be fetched",
            },
            "fields": {
                "in": "query",
                "type": "string",
                "description": "comma-delimited list-as-string of study fields to return",
            },
            "dedupe_status": {
                "in": "query",
                "type": "string",
                "enum": constants.DEDUPE_STATUSES,
                "description": "filter studies to only those with matching deduplication statuses",
            },
            "citation_status": {
                "in": "query",
                "type": "string",
                "enum": constants.USER_SCREENING_STATUSES,
                "description": "filter studies to only those with matching citation statuses",
            },
            "fulltext_status": {
                "in": "query",
                "type": "string",
                "enum": constants.USER_SCREENING_STATUSES,
                "description": "filter studies to only those with matching fulltext statuses",
            },
            "data_extraction_status": {
                "in": "query",
                "type": "string",
                "enum": constants.EXTRACTION_STATUSES,
                "description": "filter studies to only those with matching data extraction statuses",
            },
            "num_citation_reviewers": {
                "in": "query",
                "type": "integer",
                "description": "filter studies to only those with a matching number of citation reviewers",
            },
            "num_fulltext_reviewers": {
                "in": "query",
                "type": "integer",
                "description": "filter studies to only those with a matching number of fulltext reviewers",
            },
            "tag": {
                "in": "query",
                "type": "string",
                "description": "filter studies to only those with a matching (user-assigned) tag",
            },
            "tsquery": {
                "in": "query",
                "type": "string",
                "description": "filter studies to only those whose text content contains this word or phrase",
            },
            "order_by": {
                "in": "query",
                "type": "string",
                "enum": ["recency", "relevance"],
                "description": "order matching studies by either date imported or expected relevance",
            },
            "order_dir": {
                "in": "query",
                "type": "string",
                "enum": ["ASC", "DESC"],
                "description": "direction of ordering, either in ascending or descending order",
            },
            "page": {
                "in": "query",
                "type": "integer",
                "description": "page number of the collection of ordered, matching studies, starting at 0",
            },
            "per_page": {
                "in": "query",
                "type": "integer",
                "description": "number of studies to include per page",
            },
        },
        responses={
            200: "successfully got matching study record(s)",
            403: "current app user forbidden to get studies for this review",
            404: "no review with matching id was found",
        },
    )
    @use_kwargs(
        {
            "review_id": ma_fields.Int(
                required=True, validate=Range(min=1, max=constants.MAX_INT)
            ),
            "fields": DelimitedList(
                ma_fields.String(), delimiter=",", load_default=None
            ),
            "dedupe_status": ma_fields.String(
                load_default=None, validate=OneOf(constants.DEDUPE_STATUSES)
            ),
            "citation_status": ma_fields.String(
                load_default=None, validate=OneOf(constants.USER_SCREENING_STATUSES)
            ),
            "fulltext_status": ma_fields.String(
                load_default=None, validate=OneOf(constants.USER_SCREENING_STATUSES)
            ),
            "data_extraction_status": ma_fields.String(
                load_default=None, validate=OneOf(constants.EXTRACTION_STATUSES)
            ),
            "num_citation_reviewers": ma_fields.Int(
                load_default=None, validate=Range(min=1, max=3)
            ),
            "num_fulltext_reviewers": ma_fields.Int(
                load_default=None, validate=Range(min=1, max=3)
            ),
            "tag": ma_fields.String(load_default=None, validate=Length(max=25)),
            "tsquery": ma_fields.String(load_default=None, validate=Length(max=50)),
            "order_by": ma_fields.String(
                load_default="relevance", validate=OneOf(["recency", "relevance"])
            ),
            "order_dir": ma_fields.String(
                load_default="DESC", validate=OneOf(["ASC", "DESC"])
            ),
            "page": ma_fields.Int(load_default=0, validate=Range(min=0)),
            "per_page": ma_fields.Int(
                load_default=25, validate=OneOf([1, 10, 25, 50, 100, 5000])
            ),
        },
        location="query",
    )
    @jwtext.jwt_required()
    def get(
        self,
        review_id,
        fields,
        dedupe_status,
        citation_status,
        fulltext_status,
        data_extraction_status,
        num_citation_reviewers,
        num_fulltext_reviewers,
        tag,
        tsquery,
        order_by,
        order_dir,
        page,
        per_page,
    ):
        """get study record(s) for one or more matching studies"""
        current_user = jwtext.get_current_user()
        review = db.session.get(models.Review, review_id)
        if not review:
            return not_found_error(f"<Review(id={review_id})> not found")
        if not _is_allowed(current_user, review_id):
            return forbidden_error(
                f"{current_user} forbidden to get studies from this review"
            )
        if fields and "id" not in fields:
            fields.append("id")

        from ...models import Screening, Study

        stmt = sa.select(Study).where(Study.review_id == review_id)

        if dedupe_status is not None:
            stmt = stmt.where(Study.dedupe_status == dedupe_status)
        if num_citation_reviewers is not None:
            stmt = stmt.where(Study.num_citation_reviewers == num_citation_reviewers)
        if num_fulltext_reviewers is not None:
            stmt = stmt.where(Study.num_fulltext_reviewers == num_fulltext_reviewers)
        if tag:
            stmt = stmt.where(Study.tags.any_() == tag)

        if citation_status is not None:
            if citation_status in {"conflict", "excluded", "included"}:
                stmt = stmt.where(Study.citation_status == citation_status)
            elif citation_status == "pending":
                user_screened_sq = (
                    sa.select(Screening.study_id).where(
                        Screening.stage == "citation",
                        Screening.user_id == current_user.id,
                    )
                ).subquery("user_screened")
                study_ids_sq = (
                    sa.select(Study.id)
                    .where(
                        Study.dedupe_status == "not_duplicate",
                        Study.citation_status.not_in(
                            ["included", "excluded", "conflict"]
                        ),
                    )
                    .join(
                        user_screened_sq,
                        Study.id == user_screened_sq.c.study_id,
                        isouter=True,
                    )
                ).where(
                    sa.or_(
                        Study.citation_status == "not_screened",
                        user_screened_sq.c.study_id == None,
                    )
                )
                stmt = stmt.where(Study.id.in_(study_ids_sq))
            elif citation_status == "awaiting_coscreener":
                user_screened_sq = (
                    sa.select(Screening.study_id).where(
                        Screening.stage == "citation",
                        Screening.user_id == current_user.id,
                    )
                ).subquery("user_screened")
                study_ids_sq = (
                    sa.select(Study.id)
                    .where(Study.citation_status == "screened_once")
                    .join(
                        user_screened_sq,
                        Study.id == user_screened_sq.c.study_id,
                        isouter=True,
                    )
                ).where(user_screened_sq.c.study_id != None)
                stmt = stmt.where(Study.id.in_(study_ids_sq))

        if fulltext_status is not None:
            if fulltext_status in {"conflict", "excluded", "included"}:
                stmt = stmt.where(Study.fulltext_status == fulltext_status)
            elif fulltext_status == "pending":
                user_screened_sq = (
                    sa.select(Screening.study_id).where(
                        Screening.stage == "fulltext",
                        Screening.user_id == current_user.id,
                    )
                ).subquery("user_screened")
                study_ids_sq = (
                    sa.select(Study.id)
                    .where(
                        Study.citation_status == "included",
                        Study.fulltext_status.not_in(
                            ["included", "excluded", "conflict"]
                        ),
                    )
                    .join(
                        user_screened_sq,
                        Study.id == user_screened_sq.c.study_id,
                        isouter=True,
                    )
                ).where(
                    sa.or_(
                        Study.fulltext_status == "not_screened",
                        user_screened_sq.c.study_id == None,
                    )
                )
                stmt = stmt.where(Study.id.in_(study_ids_sq))
            elif fulltext_status == "awaiting_coscreener":
                user_screened_sq = (
                    sa.select(Screening.study_id).where(
                        Screening.stage == "fulltext",
                        Screening.user_id == current_user.id,
                    )
                ).subquery("user_screened")
                study_ids_sq = (
                    sa.select(Study.id)
                    .where(Study.fulltext_status == "screened_once")
                    .join(
                        user_screened_sq,
                        Study.id == user_screened_sq.c.study_id,
                        isouter=True,
                    )
                ).where(user_screened_sq.c.study_id != None)
                stmt = stmt.where(Study.id.in_(study_ids_sq))

        if data_extraction_status is not None:
            if data_extraction_status == "not_started":
                stmt = stmt.where(
                    Study.fulltext_status == "included",  # this is necessary!
                    Study.data_extraction_status == data_extraction_status,
                )
            else:
                stmt = stmt.where(
                    Study.data_extraction_status == data_extraction_status
                )

        if tsquery and order_by != "relevance":  # HACK...
            stmt = stmt.where(Study.citation_text_content.match(tsquery))

        # order, offset, and limit
        if order_by == "recency":
            order_by = sa.desc(Study.id) if order_dir == "DESC" else sa.asc(Study.id)
            stmt = stmt.order_by(order_by)
            stmt = stmt.offset(page * per_page).limit(per_page)
            return StudySchema(many=True, only=fields).dump(
                db.session.execute(stmt).scalars().all()
            )

        elif order_by == "relevance":
            if tsquery:
                stmt = stmt.where(Study.citation_text_content.match(tsquery))

            # get results and corresponding relevance scores
            limit = 10 * per_page
            stmt = stmt.order_by(db.func.random()).limit(limit)
            results = db.session.execute(stmt).scalars().all()
            scores = None

            # best option: we have a trained study ranker model
            study_ranker = StudyRanker(
                review_id, current_app.config["RANKER_MODELS_DIR"]
            )
            if study_ranker.model_fpath.exists():
                records = (
                    {
                        "text": result.fulltext.get("text_content")
                        or result.citation_text_content
                    }
                    for result in results
                )
                scores = study_ranker.predict_many(records, proba=True)[True]

            # # next best option: both positive and negative keyterms
            # if not scores:
            #     review_plan = review.review_plan
            #     suggested_keyterms = review_plan.suggested_keyterms
            #     if suggested_keyterms:
            #         incl_regex, excl_regex = reviewer_terms.get_incl_excl_terms_regex(
            #             review_plan.suggested_keyterms
            #         )
            #         scores = [
            #             reviewer_terms.get_incl_excl_terms_score(
            #                 incl_regex, excl_regex, result.citation_text_content
            #             )
            #             for result in results
            #         ]

            # # last option: just reviewer terms
            # if not scores:
            #     review_plan = review.review_plan
            #     keyterms = review_plan.keyterms
            #     if keyterms:
            #         keyterms_regex = reviewer_terms.get_keyterms_regex(keyterms)
            #         scores = [
            #             reviewer_terms.get_keyterms_score(
            #                 keyterms_regex, result.citation_text_content
            #             )
            #             for result in results
            #         ]

            # no model, let's just order results randomly...
            if not scores:
                scores = list(range(len(results)))
                random.shuffle(scores)

            # zip the results and scores together, sort and offset accordingly
            sorted_results = [
                result
                for result, _ in sorted(
                    zip(results, scores),
                    key=itemgetter(1),
                    reverse=False if order_dir == "ASC" else True,
                )
            ]
            offset = page * per_page
            return StudySchema(many=True, only=fields).dump(
                sorted_results[offset : offset + per_page]
            )


def _is_allowed(current_user: models.User, review_id: int) -> bool:
    is_allowed = current_user.is_admin
    is_allowed = (
        is_allowed
        or db.session.execute(
            sa.select(models.ReviewUserAssoc).filter_by(
                user_id=current_user.id, review_id=review_id
            )
        ).scalar_one_or_none()
        is not None
    )
    return is_allowed
