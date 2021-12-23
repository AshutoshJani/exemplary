from longformer3 import qaFunc
from flask import Flask, render_template, request

#Import ChatLearner
import os
import re
import sys
import tensorflow as tf
from os import path
sys.path.append(path.abspath('../chatbot'))

from settings import PROJECT_ROOT
from botpredictor import BotPredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    who = 'Longformer'
    userText = request.args.get('msg')
    response = qaFunc(userText)
    if(response == 0):
        who = 'Papaya'
        response = re.sub(r'_nl_|_np_', '\n', predictor.predict(session_id, userText)).strip()
    ret = (who, response)

    return response

if __name__ == "__main__":
    corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')
    knbs_dir = os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase')
    res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')

    with tf.Session() as sess:
        predictor = BotPredictor(sess, corpus_dir=corp_dir, knbase_dir=knbs_dir,
                                 result_dir=res_dir, result_file='basic')
        # This command UI has a single chat session only
        session_id = predictor.session_data.add_session()
        app.run()
