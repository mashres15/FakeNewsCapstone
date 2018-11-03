from flask import Flask, render_template, request
from .scrape import newsPrediction
from . import app, db
from .models import Fakenewscorpus

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

data = {'url': "NA",
        'content': "NA",
        'title': "NA",
        'authors': "NA",
        'page_rank_integer': "NA",
        'rank': "NA",
        'reaction_count': "NA",
        'comment_count': "NA",
        'share_count': "NA",
        'comment_plugin_count': "NA"
       }

@app.route("/results", methods=['GET', 'POST'])
def results():
    data = {'url': "NA",
        'content': "NA",
        'title': "NA",
        'authors': "NA",
        'page_rank_integer': "NA",
        'rank': "NA",
        'reaction_count': "NA",
        'comment_count': "NA",
        'share_count': "NA",
        'comment_plugin_count': "NA"
       }
    
    if request.method == 'POST':
        url = request.form['url']
        data = newsPrediction(url)
#        print(data) 
        predict = data['prediction'][0]
        news = Fakenewscorpus(
            url = url,
            content = data['content'],
            title = data['title'],
            authors = data['authors'],
            page_rank_integer = int(data['page_rank_integer']),
            rank = int(data['rank']),
            reaction_count = int(data['reaction_count']),
            comment_count = data['comment_count'],
            share_count = int(data['share_count']),
#            comment_plugin_count = data['comment_plugin_count'],
            prediction = int(predict)
        )
        exists = db.session.query(db.exists().where(Fakenewscorpus.url == url)).scalar()
        if not exists:
            db.session.add(news)
            db.session.commit()
        return render_template('results.html', data = data)
        
    return render_template('results.html', data = data)

if __name__ == "__main__":
    app.run(debug=True)