from flask import current_app, render_template
from flask_restx import Resource

from marshmallow import fields as ma_fields
from marshmallow.validate import Email
from webargs.flaskparser import use_args, use_kwargs

from colandr import api_
from ...models import db, User
from ...tasks import send_email
from ..errors import forbidden_error, not_found_error, validation_error
from ..registration import confirm_token, generate_confirmation_token
from ..schemas import UserSchema


ns = api_.namespace(
    'password reset', path='/reset',
    description='reset a user\'s password')


@ns.route('')
@ns.doc(
    summary='reset a user\'s password by sending an email',
    produces=['application/json'],
    )
class PasswordResetResource(Resource):

    @ns.doc(
        params={
            'email': {'in': 'query', 'type': 'string', 'required': True,
                      'description': 'email of user whose password is to be reset'},
            'server_name': {'in': 'query', 'type': 'string', 'default': None,
                            'description': 'name of server used to build confirmation url, e.g. "http://www.colandrapp.com"'},
            'test': {'in': 'query', 'type': 'boolean', 'default': False,
                     'description': 'if True, request will be validated but no data will be affected'},
            },
        responses={
            200: 'user was created (or would have been created if test had been False)',
            401: 'current app user not authorized to create user',
            }
        )
    @use_kwargs({
        'email': ma_fields.Str(
            required=True, validate=Email()),
        'server_name': ma_fields.Str(missing=None),
        'test': ma_fields.Boolean(missing=False)
        })
    def post(self, email, server_name, test):
        """reset user's password"""
        user = db.session.query(User).filter_by(email=email).one_or_none()
        if user is None:
            html = render_template(
                'emails/password_reset_invalid_email.html',
                email=email)
            if test is False:
                send_email.apply_async(
                    args=[[email], 'Reset Password?', '', html])
        else:
            token = generate_confirmation_token(user.email)
            if server_name:
                confirm_url = server_name + '{}/{}'.format(ns.path, token)
            else:
                confirm_url = api_.url_for(
                    ConfirmPasswordResetResource, token=token, _external=True)
            html = render_template(
                'emails/password_reset.html',
                username=user.name, confirm_url=confirm_url)
            if test is False:
                send_email.apply_async(
                    args=[[email], 'Reset Password', '', html])
                current_app.logger.info('password reset email sent to %s', email)


@ns.route('/<token>')
@ns.doc(
    summary='confirm a user\'s password reset via emailed token',
    produces=['application/json'],
    )
class ConfirmPasswordResetResource(Resource):

    @ns.doc(
        responses={
            200: 'password reset successfully confirmed',
            404: 'no user found with given email',
            422: 'invalid or expired password reset link',
            }
        )
    @use_kwargs({'token': ma_fields.String(required=True, location='view_args')})
    def get(self, token):
        """confirm a user's password reset via emailed token"""
        try:
            email = confirm_token(token)
        except Exception:
            return validation_error('the password reset link is invalid or has expired')
        user = db.session.query(User).filter_by(email=email).one_or_none()
        if not user:
            return not_found_error('no user found with email = "{}"'.format(email))
        if user.is_confirmed is False:
            return forbidden_error('user not confirmed! please first confirm your email address.')
        current_app.logger.info('password reset confirmed by %s', email)

    @ns.doc(
        params={
            'password': {'in': 'body', 'type': 'string', 'required': True,
                         'description': 'new user password to be set; min/max length requirements enforced'},
            'test': {'in': 'query', 'type': 'boolean', 'default': False,
                     'description': 'if True, request will be validated but no data will be affected'},
            },
        responses={
            200: 'new user password successfully set',
            404: 'no user found with given email'
            }
        )
    @use_args(UserSchema(only=['password']))
    @use_kwargs({
        'token': ma_fields.String(required=True, location='view_args'),
        'test': ma_fields.Boolean(missing=False)
        })
    def put(self, args, token, test):
        """set new user password after confirming reset request"""
        email = confirm_token(token, max_age=None)
        user = db.session.query(User).filter_by(email=email).one_or_none()
        # this should never happen, but *just to be safe*
        if not user:
            return not_found_error('no user found with email = "{}"'.format(email))
        user.password = User.hash_password(args['password'])
        if test is False:
            db.session.commit()
            current_app.logger.info('modified %s', user)
        else:
            db.session.rollback()
        return UserSchema().dump(user).data
