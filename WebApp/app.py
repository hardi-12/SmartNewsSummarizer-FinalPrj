from flask import Flask,render_template, request, redirect, url_for
import json
from model import summarize

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/summary", methods=['POST'])
def summary():
    if request.method == 'POST':
        url = request.form['url']
        response=summarize(url)
        return render_template('summary.html', url=url, response=response)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)