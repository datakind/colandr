from operator import itemgetter

from flask import g
from flask_restful import Resource
from flask_restful_swagger import swagger
from sqlalchemy import asc, desc, text
from sqlalchemy.sql import operators

from marshmallow import fields as ma_fields
from marshmallow.validate import OneOf, Length, Range
from webargs import missing
from webargs.fields import DelimitedList
from webargs.flaskparser import use_args, use_kwargs

from ...lib import constants
from ...lib.nlp import reviewer_terms
from ...models import db, Citation, Review
from ..errors import forbidden, no_data_found, unauthorized
from ..schemas import CitationSchema
from ..authentication import auth


class CitationResource(Resource):

    method_decorators = [auth.login_required]

    @swagger.operation()
    @use_kwargs({
        'id': ma_fields.Int(
            required=True, location='view_args',
            validate=Range(min=1, max=constants.MAX_BIGINT)),
        'fields': DelimitedList(
            ma_fields.String, delimiter=',', missing=None)
        })
    def get(self, id, fields):
        citation = db.session.query(Citation).get(id)
        if not citation:
            return no_data_found('<Citation(id={})> not found'.format(id))
        if citation.review.users.filter_by(id=g.current_user.id).one_or_none() is None:
            return unauthorized(
                '{} not authorized to get this citation'.format(g.current_user))
        return CitationSchema(only=fields).dump(citation).data

    @swagger.operation()
    @use_kwargs({
        'id': ma_fields.Int(
            required=True, location='view_args',
            validate=Range(min=1, max=constants.MAX_BIGINT)),
        'test': ma_fields.Boolean(missing=False)
        })
    def delete(self, id, test):
        citation = db.session.query(Citation).get(id)
        if not citation:
            return no_data_found('<Citation(id={})> not found'.format(id))
        if citation.review.users.filter_by(id=g.current_user.id).one_or_none() is None:
            return unauthorized(
                '{} not authorized to delete this citation'.format(g.current_user))
        if test is False:
            db.session.delete(citation)
            db.session.commit()

    @swagger.operation()
    @use_args(CitationSchema(partial=True))
    @use_kwargs({
        'id': ma_fields.Int(
            required=True, location='view_args',
            validate=Range(min=1, max=constants.MAX_BIGINT)),
        'test': ma_fields.Boolean(missing=False)
        })
    def put(self, args, id, test):
        citation = db.session.query(Citation).get(id)
        if not citation:
            return no_data_found('<Citation(id={})> not found'.format(id))
        if citation.review.users.filter_by(id=g.current_user.id).one_or_none() is None:
            return unauthorized(
                '{} not authorized to modify this citation'.format(g.current_user))
        for key, value in args.items():
            if key is missing:
                continue
            elif key == 'status':
                return forbidden('citation status can not be updated manually')
            else:
                setattr(citation, key, value)
        if test is False:
            db.session.commit()
        else:
            db.session.rollback()
        return CitationSchema().dump(citation).data


class CitationsResource(Resource):

    method_decorators = [auth.login_required]

    @swagger.operation()
    @use_kwargs({
        'review_id': ma_fields.Int(
            required=True, validate=Range(min=1, max=constants.MAX_INT)),
        'fields': DelimitedList(
            ma_fields.String(), delimiter=',', missing=None),
        'tsquery': ma_fields.String(
            missing=None, validate=Length(max=50)),
        'status': ma_fields.String(
            missing=None, validate=OneOf(['pending', 'awaiting_coscreener',
                                          'conflict', 'excluded', 'included'])),
        'tag': ma_fields.String(
            missing=None, validate=Length(max=25)),
        'order_by': ma_fields.String(
            missing='relevance', validate=OneOf(['recency', 'relevance'])),
        'order_dir': ma_fields.String(
            missing='DESC', validate=OneOf(['ASC', 'DESC'])),
        'per_page': ma_fields.Int(
            missing=25, validate=OneOf([10, 25, 50, 100, 5000])),
        'page': ma_fields.Int(
            missing=0, validate=Range(min=0)),
        })
    def get(self, review_id, fields, status, tag, tsquery,
            order_by, order_dir, per_page, page):
        review = db.session.query(Review).get(review_id)
        if not review:
            return no_data_found('<Review(id={})> not found'.format(review_id))
        if g.current_user.reviews.filter_by(id=review_id).one_or_none() is None:
            return unauthorized(
                '{} not authorized to get citations from this review'.format(
                    g.current_user))
        if fields and 'id' not in fields:
            fields.append('id')
        # build the query by components
        query = review.citations
        # filters
        if status:
            if status in ('conflict', 'excluded', 'included'):
                query = query.filter(Citation.status == status)
            elif status == 'pending':
                sql_query = """
                    SELECT t.id
                    FROM (SELECT citations.id, citations.status, screenings.user_ids
                          FROM citations
                          LEFT JOIN (SELECT citation_id, ARRAY_AGG(user_id) AS user_ids
                                     FROM citation_screenings
                                     GROUP BY citation_id
                                     ) AS screenings
                          ON citations.id = screenings.citation_id
                          ) AS t
                    WHERE
                        t.status = 'not_screened'
                        OR NOT {user_id} = ANY(t.user_ids)
                    """.format(user_id=g.current_user.id)
                query = query.filter(Citation.id.in_(text(sql_query)))
            elif status == 'awaiting_coscreener':
                sql_query = """
                    SELECT t.id
                    FROM (SELECT citations.id, citations.status, screenings.user_ids
                          FROM citations
                          LEFT JOIN (SELECT citation_id, ARRAY_AGG(user_id) AS user_ids
                                     FROM citation_screenings
                                     GROUP BY citation_id
                                     ) AS screenings
                          ON citations.id = screenings.citation_id
                          ) AS t
                    WHERE
                        t.status IN ('screened_once', 'screened_twice')
                        AND {user_id} = ANY(t.user_ids)
                    """.format(user_id=g.current_user.id)
                query = query.filter(Citation.id.in_(text(sql_query)))
        if tag:
            query = query.filter(Citation.tags.any(tag, operator=operators.eq))
        if tsquery:
            query = query.filter(Citation.text_content.match(tsquery))

        # order, offset, and limit
        if order_by == 'recency':
            order_by = desc(Citation.id) if order_dir == 'DESC' else asc(Citation.id)
            query = query.order_by(order_by)
            query = query.offset(page * per_page).limit(per_page)
            return CitationSchema(many=True, only=fields).dump(query.all()).data
        elif order_by == 'relevance':
            results = query.order_by(db.func.random()).limit(1000).all()
            review_plan = review.review_plan
            suggested_keyterms = review_plan.suggested_keyterms
            if suggested_keyterms:
                incl_regex, excl_regex = reviewer_terms.get_incl_excl_terms_regex(
                    review_plan.suggested_keyterms)
                scores = (
                    reviewer_terms.get_incl_excl_terms_score(incl_regex,
                                                              excl_regex,
                                                              result.text_content)
                    for result in results)
            else:
                keyterms_regex = reviewer_terms.get_keyterms_regex(
                    review_plan.keyterms)
                scores = (
                    reviewer_terms.get_keyterms_score(keyterms_regex,
                                                      result.text_content)
                    for result in results)
            sorted_results = [
                result for result, _
                in sorted(zip(results, scores),
                          key=itemgetter(1),
                          reverse=False if order_dir == 'ASC' else True)]
            offset = page * per_page
            return CitationSchema(many=True, only=fields).dump(
                sorted_results[offset: offset + per_page]).data

    @swagger.operation()
    @use_args(CitationSchema(partial=True))
    @use_kwargs({
        'review_id': ma_fields.Int(
            required=True, validate=Range(min=1, max=constants.MAX_INT)),
        'status': ma_fields.Str(
            missing=None, validate=OneOf(['included', 'excluded'])),
        'test': ma_fields.Boolean(missing=False)
        })
    def post(self, args, review_id, status, test):
        review = db.session.query(Review).get(review_id)
        if not review:
            return no_data_found('<Review(id={})> not found'.format(review_id))
        if g.current_user.reviews.filter_by(id=review_id).one_or_none() is None:
            return unauthorized(
                '{} not authorized to add citations to this review'.format(g.current_user))
        citation = Citation(args.pop('review_id'), **args)
        db.session.add(citation)
        if test is False:
            db.session.commit()
        else:
            db.session.rollback()
        return CitationSchema().dump(citation).data
