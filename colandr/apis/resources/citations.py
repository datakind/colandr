import flask_jwt_extended as jwtext
import sqlalchemy as sa
from flask import current_app
from flask_restx import Namespace, Resource
from marshmallow import ValidationError
from marshmallow import fields as ma_fields
from marshmallow.validate import URL, Length, OneOf, Range
from webargs import missing
from webargs.fields import DelimitedList
from webargs.flaskparser import use_args, use_kwargs

from ...extensions import db
from ...lib import constants
from ...models import DataSource, Review, Study  # Citation
from ..errors import bad_request_error, forbidden_error, not_found_error
from ..schemas import CitationSchema, DataSourceSchema
from ..swagger import citation_model


ns = Namespace(
    "citations", path="/citations", description="get, delete, update citations"
)


@ns.route("/<int:id>")
@ns.doc(
    summary="get, delete, and modify data for single citations",
    produces=["application/json"],
)
class CitationResource(Resource):
    @ns.doc(
        params={
            "fields": {
                "in": "query",
                "type": "string",
                "description": "comma-delimited list-as-string of citation fields to return",
            },
        },
        responses={
            200: "successfully got citation record",
            403: "current app user forbidden to get citation record",
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
        """get record for a single citation by id"""
        current_user = jwtext.get_current_user()
        citation = db.session.get(Citation, id)
        if not citation:
            return not_found_error(f"<Citation(id={id})> not found")
        if (
            current_user.is_admin is False
            and citation.review.review_user_assoc.filter_by(
                user_id=current_user.id
            ).one_or_none()
            is None
        ):
            return forbidden_error(f"{current_user} forbidden to get this citation")
        if fields and "id" not in fields:
            fields.append("id")
        current_app.logger.debug("got %s", citation)
        return CitationSchema(only=fields).dump(citation)

    @ns.doc(
        responses={
            204: "successfully deleted citation record",
            403: "current app user forbidden to delete citation record",
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
        """delete record for a single citation by id"""
        current_user = jwtext.get_current_user()
        citation = db.session.get(Citation, id)
        if not citation:
            return not_found_error(f"<Citation(id={id})> not found")
        if (
            current_user.is_admin is False
            and citation.review.review_user_assoc.filter_by(
                user_id=current_user.id
            ).one_or_none()
            is None
        ):
            return forbidden_error(f"{current_user} forbidden to delete this citation")
        db.session.delete(citation)
        db.session.commit()
        current_app.logger.info("deleted %s", citation)
        return "", 204

    @ns.doc(
        expect=(citation_model, "citation data to be modified"),
        responses={
            200: "citation data was modified",
            403: "current app user forbidden to modify citation",
            404: "no citation with matching id was found",
        },
    )
    @use_args(CitationSchema(partial=True), location="json")
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
        """modify record for a single citation by id"""
        current_user = jwtext.get_current_user()
        citation = db.session.get(Citation, id)
        if not citation:
            return not_found_error(f"<Citation(id={id})> not found")
        if (
            current_user.is_admin is False
            and citation.review.review_user_assoc.filter_by(
                user_id=current_user.id
            ).one_or_none()
            is None
        ):
            return forbidden_error(f"{current_user} forbidden to modify this citation")
        for key, value in args.items():
            if key is missing or key == "other_fields":
                continue
            else:
                setattr(citation, key, value)
        db.session.commit()
        current_app.logger.info("modified %s", citation)
        return CitationSchema().dump(citation)


@ns.route("")
@ns.doc(
    summary="create a single citation",
    produces=["application/json"],
)
class CitationsResource(Resource):
    @ns.doc(
        params={
            "review_id": {
                "in": "query",
                "type": "integer",
                "required": True,
                "description": "unique identifier for review for which a citation will be created",
            },
            "source_type": {
                "in": "query",
                "type": "string",
                "enum": ["database", "gray literature"],
                "description": "type of source through which citation was found",
            },
            "source_name": {
                "in": "query",
                "type": "string",
                "description": "name of source through which citation was found",
            },
            "source_url": {
                "in": "query",
                "type": "string",
                "format": "url",
                "description": "url of source through which citation was found",
            },
            "status": {
                "in": "query",
                "type": "string",
                "enum": ["not_screened", "included", "excluded"],
                "description": "known screening status of citation, if anything",
            },
        },
        expect=(citation_model, "citation data to be created"),
        responses={
            200: "successfully created citation record",
            403: "current app user forbidden to create citation for this review",
            404: "no review with matching id was found",
        },
    )
    @use_args(CitationSchema(partial=True), location="json")
    @use_kwargs(
        {
            "review_id": ma_fields.Int(
                required=True, validate=Range(min=1, max=constants.MAX_INT)
            ),
            "source_type": ma_fields.Str(
                required=True, validate=OneOf(["database", "gray literature"])
            ),
            "source_name": ma_fields.Str(load_default=None, validate=Length(max=100)),
            "source_url": ma_fields.Str(
                load_default=None, validate=[URL(relative=False), Length(max=500)]
            ),
            "status": ma_fields.Str(
                load_default=None,
                validate=OneOf(["not_screened", "included", "excluded"]),
            ),
        },
        location="query",
    )
    @jwtext.jwt_required()
    def post(self, args, review_id, source_type, source_name, source_url, status):
        """create a single citation"""
        current_user = jwtext.get_current_user()
        review = db.session.get(Review, review_id)
        if not review:
            return not_found_error(f"<Review(id={review_id})> not found")
        if (
            current_user.is_admin is False
            and current_user.review_user_assoc.filter_by(
                review_id=review_id
            ).one_or_none()
            is None
        ):
            return forbidden_error(
                f"{current_user} forbidden to add citations to this review"
            )
        # upsert the data source
        try:
            DataSourceSchema().validate(
                {
                    "source_type": source_type,
                    "source_name": source_name,
                    "source_url": source_url,
                }
            )
        except ValidationError as e:
            return bad_request_error(e.messages)
        data_source = db.session.execute(
            sa.select(DataSource).filter_by(
                source_type=source_type, source_name=source_name
            )
        ).scalar_one_or_none()
        if data_source is None:
            data_source = DataSource(source_type, source_name, source_url=source_url)
            db.session.add(data_source)
        db.session.commit()
        current_app.logger.info("inserted %s", data_source)

        # add the study
        study = Study(
            **{
                "user_id": current_user.id,
                "review_id": review_id,
                "data_source_id": data_source.id,
            }
        )
        if status is not None:
            study.citation_status = status
        db.session.add(study)
        db.session.commit()

        # *now* add the citation
        citation = args
        citation["review_id"] = review_id
        citation = CitationSchema().load(citation)  # this sanitizes the data
        assert isinstance(citation, dict)  # type guard
        citation = Citation(study.id, **citation)
        db.session.add(citation)
        db.session.commit()
        current_app.logger.info("inserted %s", citation)

        # TODO: what about deduplication?!
        # TODO: what about adding *multiple* citations via this endpoint?

        return CitationSchema().dump(citation)
