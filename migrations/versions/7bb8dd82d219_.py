"""empty message

Revision ID: 7bb8dd82d219
Revises: 0c4b878c4312
Create Date: 2016-09-23 21:29:35.032512

"""

# revision identifiers, used by Alembic.
revision = '7bb8dd82d219'
down_revision = '0c4b878c4312'

from alembic import op
import sqlalchemy as sa


def upgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('fulltexts', 'content',
               existing_type=sa.TEXT(),
               nullable=True)
    op.alter_column('fulltexts', 'filename',
               type_=sa.Unicode(length=30),
               nullable=False)
    ### end Alembic commands ###


def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('fulltexts', 'filename',
               type_=sa.UnicodeText(),
               nullable=True)
    op.alter_column('fulltexts', 'content',
               existing_type=sa.TEXT(),
               nullable=False)
    ### end Alembic commands ###
