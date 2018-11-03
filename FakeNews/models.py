from . import db
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime

class Fakenewscorpus(db.Model):
    __tablename__ = 'Fakenewscorpus'

    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String())
    content = db.Column(db.String())
    title = db.Column(db.String())
    authors = db.Column(db.String())
    page_rank_integer = db.Column(db.Integer())
    rank = db.Column(db.BigInteger())
    reaction_count = db.Column(db.BigInteger())
    comment_count = db.Column(db.BigInteger())
    share_count = db.Column(db.BigInteger())
    prediction = db.Column(db.Integer())
    entry_date = db.Column(db.DateTime(), default=datetime.utcnow)

    def __init__(self, url, content, title, authors, page_rank_integer, rank, reaction_count, comment_count, share_count, prediction):
        self.url = url
        self.content = content
        self.title = title
        self.authors = authors
        self.page_rank_integer = page_rank_integer
        self.rank = rank
        self.reaction_count = reaction_count
        self.comment_count = comment_count
        self.share_count = share_count
        self.prediction = prediction

    def __repr__(self):
        return '<id {}>'.format(self.id)