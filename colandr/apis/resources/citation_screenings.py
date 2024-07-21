import random

import flask_jwt_extended as jwtext
import sqlalchemy as sa
from flask import current_app
from flask_restx import Namespace, Resource
from marshmallow import fields as ma_fields
from marshmallow.validate import Range
from webargs import missing
from webargs.fields import DelimitedList
from webargs.flaskparser import use_args, use_kwargs

from ... import models, tasks
from ...extensions import db
from ...lib import constants
from ...utils import assign_status
from .. import auth
from ..errors import bad_request_error, forbidden_error, not_found_error
from ..schemas import ScreeningSchema, ScreeningV2Schema
from ..swagger import screening_model


ns = Namespace(
    "citation_screenings",
    path="/citations",
    description="get, create, delete, modify citation screenings",
)


@ns.route("/<int:id>/screenings")
@ns.doc(
    summary="get, create, delete, and modify data for a single citations's screenings",
    produces=["application/json"],
)
class CitationScreeningsResource(Resource):
    @ns.doc(
        params={
            "fields": {
                "in": "query",
                "type": "string",
                "description": "comma-delimited list-as-string of screening fields to return",
            },
        },
        responses={
            200: "successfully got citation screening record(s)",
            403: "current app user forbidden to get citation screening record(s)",
            404: "no citation with matching id was found",
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
    @use_kwargs(
        {"fields": DelimitedList(ma_fields.String, delimiter=",", load_default=None)},
        location="query",
    )
    @jwtext.jwt_required()
    def get(self, id, fields):
        """get screenings for a single citation by id"""
        current_user = jwtext.get_current_user()
        # check current user authorization
        study = db.session.get(models.Study, id)
        if not study:
            return not_found_error(f"<Study(id={id})> not found")
        if not _is_allowed(current_user, study.review_id):
            return forbidden_error(
                f"{current_user} forbidden to get citation screenings for this review"
            )
        screenings = db.session.execute(
            study.screenings.select().filter_by(stage="citation")
        ).scalars()
        if not screenings:
            return not_found_error(f"no screenings for <Study(id={id})> found")
        # HACK: hide the consolidated (v2) screening schema from this api
        if fields:
            if "id" not in fields:
                fields.append("id")
            if "citation_id" in fields:
                fields.append("study_id")
                fields.remove("citation_id")
        screenings_dumped = [
            _convert_screening_v2_into_v1(record)
            for record in ScreeningV2Schema(many=True, only=fields).dump(screenings)
        ]
        return screenings_dumped

    @ns.doc(
        responses={
            204: "successfully deleted citation screening record",
            403: "current app user forbidden to delete citation screening record",
            404: "no citation with matching id was found",
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
        """delete current app user's screening for a single citation by id"""
        current_user = jwtext.get_current_user()
        # check current user authorization
        study = db.session.get(models.Study, id)
        if not study:
            return not_found_error(f"<Study(id={id})> not found")
        if not _is_allowed(current_user, study.review_id):
            return forbidden_error(
                f"{current_user} forbidden to delete citation screening for this review"
            )

        screening = db.session.execute(
            study.screenings.select().filter_by(
                stage="citation", user_id=current_user.id
            )
        ).scalar_one_or_none()
        if not screening:
            return forbidden_error(
                f"{current_user} has not screened {study}, so nothing to delete"
            )
        db.session.delete(screening)
        db.session.commit()
        current_app.logger.info("deleted %s", screening)
        return "", 204

    @ns.doc(
        expect=(screening_model, "citation screening record to be created"),
        responses={
            200: "citation screening record was created",
            403: "current app user forbidden to create citation screening",
            404: "no citation with matching id was found",
            422: "invalid citation screening record",
        },
    )
    @use_args(ScreeningSchema(partial=["user_id", "review_id"]), location="json")
    @use_kwargs(
        {
            "id": ma_fields.Int(
                required=True, validate=Range(min=1, max=constants.MAX_BIGINT)
            ),
        },
        location="view_args",
    )
    @jwtext.jwt_required()
    def post(self, args, id):
        """create a screening for a single citation by id"""
        current_user = jwtext.get_current_user()
        # check current user authorization
        study = db.session.get(models.Study, id)
        if not study:
            return not_found_error(f"<Study(id={id})> not found")
        if not _is_allowed(current_user, study.review_id):
            return forbidden_error(
                f"{current_user} forbidden to screen citations for this review"
            )
        # validate and add screening
        if args["status"] == "excluded" and not args["exclude_reasons"]:
            return bad_request_error("screenings that exclude must provide a reason")
        if current_user.is_admin:
            if "user_id" not in args:
                return bad_request_error(
                    "admins must specify 'user_id' when creating a citation screening"
                )
            else:
                user_id = args["user_id"]
        else:
            user_id = current_user.id

        if db.session.execute(
            study.screenings.select().filter_by(
                stage="citation", user_id=current_user.id
            )
        ).one_or_none():
            return forbidden_error(f"{current_user} has already screened {study}")

        screening = models.Screening(
            user_id=user_id,
            review_id=study.review_id,
            study_id=id,
            stage="citation",
            status=args["status"],
            exclude_reasons=args["exclude_reasons"],
        )  # type: ignore
        study.screenings.add(screening)
        db.session.commit()
        current_app.logger.info("inserted %s", screening)
        tasks.train_study_ranker_model.apply_async(args=[study.review_id, screening.id])
        return _convert_screening_v2_into_v1(ScreeningV2Schema().dump(screening))

    @ns.doc(
        expect=(screening_model, "citation screening data to be modified"),
        responses={
            200: "citation screening data was modified",
            401: "current app user not authorized to modify citation screening",
            404: "no citation with matching id was found, or no citation screening exists for current app user",
            422: "invalid modified citation screening data",
        },
    )
    @use_args(
        ScreeningSchema(
            only=["user_id", "status", "exclude_reasons"],
            partial=["exclude_reasons"],
        ),
        location="json",
    )
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
        """modify current app user's screening of a single citation by id"""
        current_user = jwtext.get_current_user()
        study = db.session.get(models.Study, id)
        if not study:
            return not_found_error(f"<Study(id={id})> not found")
        if current_user.is_admin is True and "user_id" in args:
            screening = db.session.execute(
                study.screenings.select().filter_by(
                    stage="citation", user_id=args["user_id"]
                )
            ).scalar_one_or_none()
        else:
            screening = db.session.execute(
                study.screenings.select().filter_by(
                    stage="citation", user_id=current_user.id
                )
            ).scalar_one_or_none()
        if not screening:
            return not_found_error(f"{current_user} has not screened this citation")
        if args["status"] == "excluded" and not args.get("exclude_reasons"):
            return bad_request_error("screenings that exclude must provide a reason")
        for key, value in args.items():
            if key is missing:
                continue
            else:
                setattr(screening, key, value)
        db.session.commit()
        current_app.logger.debug("modified %s", screening)
        return _convert_screening_v2_into_v1(ScreeningV2Schema().dump(screening))


@ns.route("/screenings")
@ns.doc(
    summary="get one or many citation screenings",
    produces=["application/json"],
)
class CitationsScreeningsResource(Resource):
    @ns.doc(
        params={
            "citation_id": {
                "in": "query",
                "type": "integer",
                "description": "unique identifier of citation for which to get all citation screenings",
            },
            "user_id": {
                "in": "query",
                "type": "integer",
                "description": "unique identifier of user for which to get all citation screenings",
            },
            "review_id": {
                "in": "query",
                "type": "integer",
                "description": "unique identifier of review for which to get citation screenings",
            },
            "status_counts": {
                "in": "query",
                "type": "boolean",
                "default": False,
                "description": "if True, group screenings by status and return the counts; if False, return the screening records themselves",
            },
        },
        responses={
            200: "successfully got citation screening record(s)",
            400: "bad request: citation_id, user_id, or review_id required",
            403: "current app user forbidden to get citation screening record(s)",
            404: "no citation with matching id was found",
        },
    )
    @use_kwargs(
        {
            "citation_id": ma_fields.Int(
                load_default=None, validate=Range(min=1, max=constants.MAX_BIGINT)
            ),
            "user_id": ma_fields.Int(
                load_default=None, validate=Range(min=1, max=constants.MAX_INT)
            ),
            "review_id": ma_fields.Int(
                load_default=None, validate=Range(min=1, max=constants.MAX_INT)
            ),
            "status_counts": ma_fields.Boolean(load_default=False),
        },
        location="query",
    )
    @jwtext.jwt_required()
    def get(self, citation_id, user_id, review_id, status_counts):
        """get all citation screenings by citation, user, or review id"""
        current_user = jwtext.get_current_user()
        if not any([citation_id, user_id, review_id]):
            return bad_request_error(
                "citation, user, and/or review id must be specified"
            )

        stmt = (
            sa.select(models.Screening)
            if status_counts is False
            else sa.select(models.Screening.status, db.func.count(1))
        )
        stmt = stmt.where(models.Screening.stage == "citation")
        if citation_id is not None:
            # check user authorization
            study = db.session.get(models.Study, citation_id)
            if not study:
                return not_found_error(f"<Study(id={citation_id})> not found")
            if not _is_allowed(current_user, study.review_id):
                return forbidden_error(
                    f"{current_user} forbidden to get screenings for {study}"
                )
            stmt = stmt.where(models.Screening.study_id == citation_id)
        if user_id is not None:
            # check user authorization
            user = db.session.get(models.User, user_id)
            if not user:
                return not_found_error(f"<User(id={user_id})> not found")
            if current_user.is_admin is False and not any(
                user_id == user.id
                for review in current_user.reviews
                for user in review.users
            ):
                return forbidden_error(
                    f"{current_user} forbidden to get screenings for {user}"
                )
            stmt = stmt.where(models.Screening.user_id == user_id)
        if review_id is not None:
            # check user authorization
            review = db.session.get(models.Review, review_id)
            if not review:
                return not_found_error(f"<Review(id={review_id})> not found")
            if (
                current_user.is_admin is False
                and db.session.execute(
                    review.review_user_assoc.select().filter_by(user_id=current_user.id)
                ).one_or_none()
                is None
            ):
                return forbidden_error(
                    f"{current_user} forbidden to get screenings for {review}"
                )
            stmt = stmt.where(models.Screening.review_id == review_id)

        if status_counts is True:
            stmt = stmt.group_by(models.Screening.status)
            return {row.status: row.count for row in db.session.execute(stmt)}
        else:
            results = db.session.execute(stmt).scalars()
            return [
                _convert_screening_v2_into_v1(record)
                for record in ScreeningV2Schema(partial=True, many=True).dump(results)
            ]

    @ns.doc(
        params={
            "review_id": {
                "in": "query",
                "type": "integer",
                "required": True,
                "description": "unique identifier of review for which to create citation screenings",
            },
            "user_id": {
                "in": "query",
                "type": "integer",
                "description": "unique identifier of user screening citations, if not current app user",
            },
        },
        expect=([screening_model], "citation screening records to create"),
        responses={
            200: "successfully created citation screening record(s)",
            403: "current app user forbidden to create citation screening records",
            404: "no review with matching id was found",
        },
    )
    @use_args(
        ScreeningSchema(many=True, partial=["user_id", "review_id"]), location="json"
    )
    @use_kwargs(
        {
            "review_id": ma_fields.Int(
                required=True, validate=Range(min=1, max=constants.MAX_INT)
            ),
            "user_id": ma_fields.Int(
                load_default=None, validate=Range(min=1, max=constants.MAX_INT)
            ),
        },
        location="query",
    )
    @auth.jwt_admin_required()
    def post(self, args, review_id, user_id):
        """create one or more citation screenings (ADMIN ONLY)"""
        current_user = jwtext.get_current_user()
        review = db.session.get(models.Review, review_id)
        if not review:
            return not_found_error(f"<Review(id={review_id})> not found")
        # bulk insert citation screenings
        screener_user_id = user_id or current_user.id
        screenings_to_insert = []
        for screening in args:
            screening["user_id"] = screener_user_id
            screening["review_id"] = review_id
            screening["stage"] = "citation"
            # HACK to account for citation+fulltext screening consolidation
            if "citation_id" in screening:
                screening["study_id"] = screening.pop("citation_id")
            screenings_to_insert.append(screening)
        db.session.execute(sa.insert(models.Screening), screenings_to_insert)
        db.session.commit()
        current_app.logger.info(
            "inserted %s citation screenings", len(screenings_to_insert)
        )
        # bulk update citation statuses
        study_ids: list[int] = sorted(s["study_id"] for s in screenings_to_insert)
        study_num_citation_reviewers: list[int] = random.choices(
            [num_pct["num"] for num_pct in review.citation_reviewer_num_pcts],
            weights=[num_pct["pct"] for num_pct in review.citation_reviewer_num_pcts],
            k=len(study_ids),
        )
        results = db.session.execute(
            sa.select(
                models.Screening.study_id, sa.func.array_agg(models.Screening.status)
            )
            .where(models.Screening.stage == "citation")
            .where(models.Screening.study_id == sa.any_(study_ids))
            .group_by(models.Screening.study_id)
            .order_by(models.Screening.study_id)
        )
        studies_to_update = [
            {"id": row[0], "citation_status": assign_status(row[1], num_reviewers)}
            for row, num_reviewers in zip(results, study_num_citation_reviewers)
        ]
        db.session.execute(sa.update(models.Study), studies_to_update)
        db.session.commit()
        current_app.logger.info(
            "updated citation_status for %s studies", len(studies_to_update)
        )
        # get include/exclude counts on review
        status_counts = review.num_citations_by_status(["included", "excluded"])
        n_included = status_counts.get("included", 0)
        n_excluded = status_counts.get("excluded", 0)
        # do we have to suggest keyterms?
        if n_included >= 25 and n_excluded >= 25:
            sample_size = min(n_included, n_excluded)
            tasks.suggest_keyterms.apply_async(args=[review_id, sample_size])
        # do we have to train a ranking model?
        if n_included >= 100 and n_excluded >= 100:
            tasks.train_citation_ranking_model.apply_async(
                args=[review_id], countdown=3
            )


def _convert_screening_v2_into_v1(record) -> dict:
    # remove stage field, if present
    record.pop("stage", None)
    # rename study_id field to citation_id
    try:
        record["citation_id"] = record.pop("study_id")
    except KeyError:
        pass
    return record


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
