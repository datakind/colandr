import logging
import os

from dotenv import load_dotenv


# load `.env` file based on `.env.example` containing config values
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(basedir, ".env"))


class Config:
    TESTING = False
    SECRET_KEY = os.environ["COLANDR_SECRET_KEY"]
    MAX_CONTENT_LENGTH = 40 * 1024 * 1024  # 40MB file upload limit

    PASSWORD_SALT = os.environ.get("COLANDR_PASSWORD_SALT")  # TODO: remove this
    # SSL_DISABLE = False
    APP_URL_DOMAIN = "http://localhost:5001/api"
    LOG_LEVEL = os.getenv("COLANDR_LOG_LEVEL", logging.INFO)

    # celery+redis config
    CELERY = {
        "broker_url": os.environ.get(
            "COLANDR_CELERY_BROKER_URL", "redis://localhost:6379/0"
        ),
        "result_backend": os.environ.get(
            "COLANDR_CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
        ),
        "accept_content": ["json"],
        "task_serializer": "json",
        "result_serializer": "json",
        # ref: https://steve.dignam.xyz/2023/05/20/many-problems-with-celery
        "worker_prefetch_multiplier": 1,
        "task_acks_late": True,
    }

    # cache config
    CACHE_TYPE = "SimpleCache"
    # TODO: figure out if/how we want to use redis for caching
    # CACHE_TYPE = "RedisCache",
    # CACHE_REDIS_HOST = os.environ.get("COLANDR_REDIS_HOST", "localhost")

    # sql db config
    SQLALCHEMY_DATABASE_URI = (
        "postgresql://"
        f"{os.environ['COLANDR_DB_USER']}:{os.environ['COLANDR_DB_PASSWORD']}"
        "@colandr-db:5432"
        f"/{os.environ['COLANDR_DB_NAME']}"
    )
    # TODO: decide if we'd rather just use a single env var for this
    # at the moment, it seems like "pdfestrian" may require this
    # SQLALCHEMY_DATABASE_URI = os.environ["COLANDR_DATABASE_URI"]
    SQLALCHEMY_ECHO = False

    # authentication config
    JWT_ACCESS_LIFESPAN = {"hours": 3}
    JWT_REFRESH_LIFESPAN = {"days": 7}

    # files-on-disk config
    COLANDR_APP_DIR = os.environ.get("COLANDR_APP_DIR", "/tmp")
    DEDUPE_MODELS_DIR = os.path.join(COLANDR_APP_DIR, "colandr_data", "dedupe")
    RANKING_MODELS_DIR = os.path.join(COLANDR_APP_DIR, "colandr_data", "ranking_models")
    CITATIONS_DIR = os.path.join(COLANDR_APP_DIR, "colandr_data", "citations")
    FULLTEXT_UPLOADS_DIR = os.path.join(COLANDR_APP_DIR, "colandr_data", "fulltexts")
    ALLOWED_FULLTEXT_UPLOAD_EXTENSIONS = {".txt", ".pdf"}

    # email server config
    MAIL_SERVER = os.environ.get("COLANDR_MAIL_SERVER")
    MAIL_PORT = os.environ.get("COLANDR_MAIL_PORT")
    MAIL_USE_TLS = (
        bool(int(os.environ["COLANDR_MAIL_USE_TLS"]))
        if os.environ.get("COLANDR_MAIL_USE_TLS")
        else None
    )
    MAIL_USE_SSL = (
        bool(int(os.environ["COLANDR_MAIL_USE_SSL"]))
        if os.environ.get("COLANDR_MAIL_USE_SSL")
        else None
    )
    MAIL_USERNAME = os.environ.get("COLANDR_MAIL_USERNAME")
    MAIL_PASSWORD = os.environ.get("COLANDR_MAIL_PASSWORD")
    MAIL_DEFAULT_SENDER = f"colandr <{MAIL_USERNAME}>"
    MAIL_SUBJECT_PREFIX = "[colandr]"
    MAIL_ADMINS = ["burtdewilde@gmail.com"]


class ProductionConfig(Config):
    FLASK_ENV = "production"


class DevelopmentConfig(Config):
    FLASK_ENV = "development"


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_ECHO = True


configs = {
    "dev": DevelopmentConfig,
    "test": TestingConfig,
    "prod": ProductionConfig,
    "default": ProductionConfig,
}
