"""empty message

Revision ID: 642f9119f981
Revises: 7bb8dd82d219
Create Date: 2016-09-24 22:07:21.763158

"""

# revision identifiers, used by Alembic.
revision = '642f9119f981'
down_revision = '7bb8dd82d219'

from alembic import op
import sqlalchemy as sa


def upgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('fulltexts', 'filename',
               existing_type=sa.VARCHAR(length=30),
               nullable=True)
    ### end Alembic commands ###


def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('fulltexts', 'filename',
               existing_type=sa.VARCHAR(length=30),
               nullable=False)
    ### end Alembic commands ###
