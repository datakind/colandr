import logging
import os
from time import sleep

import arrow
from flask import current_app
from flask_mail import Message
from sqlalchemy import create_engine, func, types as sqltypes
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.orm.session import Session
from sqlalchemy.sql import case, delete, exists, select, text

from . import celery, mail
from .lib.utils import load_dedupe_model, make_record_immutable
from .models import (db, Citation, DedupeBlockingMap, DedupeCoveredBlocks,
                     DedupePluralBlock, DedupePluralKey, DedupeSmallerCoverage,
                     User)


@celery.task
def send_email(recipients, subject, text_body, html_body):
    msg = Message(current_app.config['MAIL_SUBJECT_PREFIX'] + ' ' + subject,
                  sender=current_app.config['MAIL_DEFAULT_SENDER'],
                  recipients=recipients)
    msg.body = text_body
    msg.html = html_body
    mail.send(msg)


@celery.task
def remove_unconfirmed_user(email):
    user = db.session.query(User).filter_by(email=email).one_or_none()
    if user and user.is_confirmed is False:
        db.session.delete(user)
        db.session.commit()


def get_candidate_dupes(results):
    block_id = None
    records = []
    for row in results:
        if row.block_id != block_id:
            if records:
                yield records
            block_id = row.block_id
            records = []
        smaller_ids = frozenset(row.smaller_ids)
        records.append((row.citation_id,
                        make_record_immutable(dict(row)),
                        smaller_ids))
    if records:
        yield records


@celery.task
def deduplicate_citations(review_id):
    # TODO: see about setting this path in app config
    deduper = load_dedupe_model(
        os.path.join(current_app.config['DEDUPE_MODELS_FOLDER'],
                     'dedupe_citations_settings'))
    engine = create_engine(
        current_app.config['SQLALCHEMY_DATABASE_URI'],
        server_side_cursors=True, echo=False)
    with engine.connect() as conn:

        # wait until no more review citations have been created in 60+ seconds
        stmt = select([func.max(Citation.created_at)])\
            .where(Citation.review_id == review_id)
        while True:
            max_created_at = conn.execute(stmt).fetchone()[0]
            print('citation most recently created at {}'.format(max_created_at))
            if (arrow.utcnow().naive - max_created_at).total_seconds() < 60:
                sleep(15)
            else:
                break

        # if all review citations have been deduped, cancel
        stmt = select(
            [exists().where(Citation.review_id == review_id).where(Citation.deduplication == {})])
        un_deduped_citations = conn.execute(stmt).fetchone()[0]
        if un_deduped_citations is False:
            raise Exception

        # remove rows for this review
        # which we'll add back with the latest citations included
        for table in [DedupeBlockingMap, DedupePluralKey, DedupePluralBlock,
                      DedupeCoveredBlocks, DedupeSmallerCoverage]:
            stmt = delete(table).where(getattr(table, 'review_id') == review_id)
            result = conn.execute(stmt)
            rows_deleted = result.rowcount
            print('deleted {} rows from {}'.format(rows_deleted, table.__tablename__))

        # if deduper learned an Index Predicate
        # we have to take a pass through the data and create indices
        for field in deduper.blocker.index_fields:
            col_type = getattr(Citation, field).property.columns[0].type
            print('index predicate:', field, col_type)
            stmt = select([getattr(Citation, field)])\
                .where(Citation.review_id == review_id)\
                .distinct()
            results = conn.execute(stmt)
            if isinstance(col_type, sqltypes.ARRAY):
                field_data = (tuple(row[0]) for row in results)
            else:
                field_data = (row[0] for row in results)
            deduper.blocker.index(field_data, field)

        # now we're ready to write our blocking map table by creating a generator
        # that yields unique (block_key, citation_id, review_id) tuples
        stmt = select([Citation.id, Citation.title, Citation.authors,
                       Citation.pub_year.label('publication_year'),  # HACK for now
                       Citation.abstract, Citation.doi])\
            .where(Citation.review_id == review_id)
        results = conn.execute(stmt)
        data = ((row[0], make_record_immutable(dict(row)))
                for row in results)
        b_data = ((citation_id, review_id, block_key)
                  for block_key, citation_id in deduper.blocker(data))
        conn.execute(
            DedupeBlockingMap.__table__.insert(),
            [{'citation_id': row[0], 'review_id': row[1], 'block_key': row[2]}
             for row in b_data])

        # now fill review rows back in
        stmt = select([DedupeBlockingMap.review_id, DedupeBlockingMap.block_key])\
            .where(DedupeBlockingMap.review_id == review_id)\
            .group_by(DedupeBlockingMap.review_id, DedupeBlockingMap.block_key)\
            .having(func.count(1) > 1)
        conn.execute(
            DedupePluralKey.__table__.insert()\
                .from_select(['review_id', 'block_key'], stmt))

        stmt = select([DedupePluralKey.block_id,
                       DedupeBlockingMap.citation_id,
                       DedupeBlockingMap.review_id])\
            .where(DedupePluralKey.block_key == DedupeBlockingMap.block_key)\
            .where(DedupeBlockingMap.review_id == review_id)
        conn.execute(
            DedupePluralBlock.__table__.insert()\
                .from_select(['block_id', 'citation_id', 'review_id'], stmt))

        # To use Kolb, et. al's Redundant Free Comparison scheme, we need to
        # keep track of all the block_ids that are associated with particular
        # citation records
        stmt = select([DedupePluralBlock.citation_id,
                       DedupePluralBlock.review_id,
                       func.array_agg(aggregate_order_by(DedupePluralBlock.block_id,
                                                         DedupePluralBlock.block_id.desc()),
                                      type_=sqltypes.ARRAY(sqltypes.BigInteger)).label('sorted_ids')])\
            .where(DedupePluralBlock.review_id == review_id)\
            .group_by(DedupePluralBlock.citation_id, DedupePluralBlock.review_id)
        conn.execute(
            DedupeCoveredBlocks.__table__.insert()\
                .from_select(['citation_id', 'review_id', 'sorted_ids'], stmt))

        # for every block of records, we need to keep track of a citation records's
        # associated block_ids that are SMALLER than the current block's id
        ugh = 'dedupe_covered_blocks.sorted_ids[0: array_position(dedupe_covered_blocks.sorted_ids, dedupe_plural_block.block_id) - 1] AS smaller_ids'
        stmt = select([DedupePluralBlock.citation_id,
                       DedupePluralBlock.review_id,
                       DedupePluralBlock.block_id,
                       text(ugh)])\
            .where(DedupePluralBlock.citation_id == DedupeCoveredBlocks.citation_id)\
            .where(DedupePluralBlock.review_id == review_id)
        conn.execute(
            DedupeSmallerCoverage.__table__.insert()\
                .from_select(['citation_id', 'review_id', 'block_id', 'smaller_ids'], stmt))

        # set dedupe model similarity threshold from the data
        stmt = select([Citation.id, Citation.title, Citation.authors,
                       Citation.pub_year.label('publication_year'),  # HACK for now
                       Citation.abstract, Citation.doi])\
            .where(Citation.review_id == review_id)\
            .order_by(func.random())\
            .limit(20000)
        results = conn.execute(stmt)
        dupe_threshold = deduper.threshold(
            {row.id: make_record_immutable(dict(row)) for row in results},
            recall_weight=0.5)

        # apply dedupe model to get clusters of duplicate records
        stmt = select([Citation.id.label('citation_id'), Citation.title, Citation.authors,
                       Citation.pub_year.label('publication_year'),  # HACK for now
                       Citation.abstract, Citation.doi,
                       DedupeSmallerCoverage.block_id, DedupeSmallerCoverage.smaller_ids])\
            .where(Citation.id == DedupeSmallerCoverage.citation_id)\
            .where(Citation.review_id == review_id)\
            .order_by(DedupeSmallerCoverage.block_id)
        results = conn.execute(stmt)

        clustered_dupes = deduper.matchBlocks(
            get_candidate_dupes(results),
            threshold=dupe_threshold)
        print('found {} duplicate clusters'.format(len(clustered_dupes)))

        citations_to_update = []
        for cids, scores in clustered_dupes:
            cid_scores = {int(cid): float(score) for cid, score in zip(cids, scores)}
            stmt = select([
                    Citation.id,
                    (case([(Citation.title == None, 1)]) +
                     case([(Citation.abstract == None, 1)]) +
                     case([(Citation.pub_year == None, 1)]) +
                     case([(Citation.pub_month == None, 1)]) +
                     case([(Citation.authors == {}, 1)]) +
                     case([(Citation.keywords == {}, 1)]) +
                     case([(Citation.type_of_reference == None, 1)]) +
                     case([(Citation.journal_name == None, 1)]) +
                     case([(Citation.issue_number == None, 1)]) +
                     case([(Citation.doi == None, 1)]) +
                     case([(Citation.issn == None, 1)]) +
                     case([(Citation.publisher == None, 1)]) +
                     case([(Citation.language == None, 1)])
                     ).label('n_null_cols')])\
                .where(Citation.review_id == review_id)\
                .where(Citation.id.in_([int(cid) for cid in cids]))\
                .order_by(text('n_null_cols ASC'))\
                .limit(1)
            result = conn.execute(stmt).fetchone()
            canonical_citation_id = result.id
            for cid, score in cid_scores.items():
                if cid == canonical_citation_id:
                    citations_to_update.append(
                        {'id': cid,
                         'deduplication': {'is_duplicate': False}
                         })
                else:
                    citations_to_update.append(
                        {'id': cid,
                         'status': 'excluded',
                         'deduplication': {'is_duplicate': True,
                                           'canonical_id': canonical_citation_id,
                                           'duplicate_score': score}
                         })
        session = Session(bind=conn)
        session.bulk_update_mappings(Citation, citations_to_update)
        session.commit()
