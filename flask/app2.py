#imports
from os import path
import sys
from flask import Flask, render_template, request
from longformer1 import qaFunc

# #import ChatLearner
# sys.path.append(path.abspath('../chatbot'))
# from botui import bot_ui

app = Flask(__name__)

#define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    response = qaFunc(userText)
    if response == 0:
        # response = bot_ui(userText)
        response = 0
        return response
    else:
        return response

if __name__ == "__main__":
    app.run()
