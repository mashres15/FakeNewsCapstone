import os
from flask_script import Manager, prompt_bool
from FakeNews import app, db

manager = Manager(app)

@manager.command
def initdb():
    db.create_all()
    print("Database Initialized")
    
@manager.command
def dropdb():
    if prompt_bool("Are you sure to drop the DB?"):
        db.drop_all()
        print("Database Dropped")
    
if __name__ == '__main__':
    manager.run()