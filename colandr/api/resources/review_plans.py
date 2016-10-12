from flask import g
from flask_restful import Resource
from flask_restful_swagger import swagger

from marshmallow import fields as ma_fields
from marshmallow.validate import Range
from webargs import missing
from webargs.fields import DelimitedList
from webargs.flaskparser import use_args, use_kwargs

from ...models import db, Review
from ...lib import constants
from ..errors import no_data_found, unauthorized
from ..schemas import ReviewPlanSchema
from ..authentication import auth


class ReviewPlanResource(Resource):

    method_decorators = [auth.login_required]

    @swagger.operation()
    @use_kwargs({
        'id': ma_fields.Int(
            required=True, location='view_args',
            validate=Range(min=1, max=constants.MAX_INT)),
        'fields': DelimitedList(
            ma_fields.String, delimiter=',', missing=None)
        })
    def get(self, id, fields):
        review = db.session.query(Review).get(id)
        if not review:
            return no_data_found('<Review(id={})> not found'.format(id))
        if review.users.filter_by(id=g.current_user.id).one_or_none() is None:
            return unauthorized(
                '{} not authorized to get this review plan'.format(g.current_user))
        if fields and 'id' not in fields:
            fields.append('id')
        return ReviewPlanSchema(only=fields).dump(review.review_plan).data

    # NOTE: since review plans are created automatically upon review insertion
    # and deleted automatically upon review deletion, "delete" here amounts
    # to nulling out its non-required fields
    @swagger.operation()
    @use_kwargs({
        'id': ma_fields.Int(
            required=True, location='view_args',
            validate=Range(min=1, max=constants.MAX_INT)),
        'test': ma_fields.Boolean(missing=False)
        })
    def delete(self, id, test):
        review = db.session.query(Review).get(id)
        if not review:
            return no_data_found('<Review(id={})> not found'.format(id))
        if review.owner is not g.current_user:
            return unauthorized(
                '{} not authorized to delete this review plan'.format(g.current_user))
        review_plan = review.review_plan
        review_plan.objective = ''
        review_plan.research_questions = []
        review_plan.pico = {}
        review_plan.keyterms = []
        review_plan.selection_criteria = []
        review_plan.data_extraction_form = []
        if test is False:
            # db.session.delete(review_plan)
            db.session.commit()
        else:
            db.session.rollback()
        return '', 204

    @swagger.operation()
    @use_args(ReviewPlanSchema(partial=True))
    @use_kwargs({
        'id': ma_fields.Int(
            required=True, location='view_args',
            validate=Range(min=1, max=constants.MAX_INT)),
        'test': ma_fields.Boolean(missing=False)
        })
    def put(self, args, id, test):
        review = db.session.query(Review).get(id)
        if not review:
            return no_data_found('<Review(id={})> not found'.format(id))
        if review.owner is not g.current_user:
            return unauthorized(
                '{} not authorized to create this review plan'.format(g.current_user))
        review_plan = review.review_plan
        if not review_plan:
            return no_data_found('<ReviewPlan(review_id={})> not found'.format(id))
        if review_plan.review.owner is not g.current_user:
            return unauthorized(
                '{} not authorized to update this review plan'.format(g.current_user))
        for key, value in args.items():
            if key is missing:
                continue
            else:
                setattr(review_plan, key, value)
        if test is False:
            db.session.commit()
        else:
            db.session.rollback()
        return ReviewPlanSchema().dump(review_plan).data
