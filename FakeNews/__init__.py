from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import os
import sys

#__all__ = ["scrape", "models", "customTransfomers"]

app = Flask(__name__)
POSTGRES = {
    'user': 'admin',
    'pw': 'admin',
    'db': 'fakenews',
    'host': 'localhost',
    'port': '5432',
}
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://%(user)s:\
%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False

db = SQLAlchemy(app)

from . import models
from . import views
from . import customTransfomers
from .customTransfomers import *
from . import scrape
sys.modules['customTransfomers'] = customTransfomers 