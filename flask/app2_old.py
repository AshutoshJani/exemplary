#imports
from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from longformer3 import qaFunc

app = Flask(__name__)
""" CHATTERBOT = {
        'name': 'Tech Support Bot',
        'logic_adapters': [
             {
                "import_path": "chatterbot.logic.BestMatch",
                "statement_comparison_function": "chatterbot.comparisons.levenshtein_distance",
                "response_selection_method": "chatterbot.response_selection.get_first_response"
             },

                {
                    'import_path': 'chatterbot.logic.LowConfidenceAdapter',
                    'threshold': 0.90,
                    'default_response': 'I am sorry, but I do not understand.'
                },

        ],
        'storage_adapter' : 'chatterbot.storage.SQLStorageAdapter'
} """
#create chatbot
englishBot = ChatBot(
    'Education bot',
    logic_adapters = [
             {
                "import_path": "chatterbot.logic.BestMatch",
                "statement_comparison_function": "chatterbot.comparisons.levenshtein_distance",
                "response_selection_method": "chatterbot.response_selection.get_first_response",
                'maximum_similarity_threshold': 0.75
             },          
        ],
    storage_adapter = 'chatterbot.storage.SQLStorageAdapter'
)
""" {
    'import_path': 'chatterbot.logic.LowConfidenceAdapter',
    'threshold': 0.90,
    'default_response': 'I am sorry, but I do not understand.'
    }, """
trainer = ChatterBotCorpusTrainer(englishBot)
trainer.train("chatterbot.corpus.english") #train the chatter bot for english

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
        return englishBot.get_response(userText)
    else:
        return response

if __name__ == "__main__":
    app.run()