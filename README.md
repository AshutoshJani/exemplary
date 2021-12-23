# Exemplary

## Exemplary is a chatbot with two core algorithms working in tandem to create a more robust Chatbot that can handle different types of questions. At its core Exemplary has two algorithms working in tandem:
1. [ChatLearner](https://github.com/bshao001/ChatLearner)
2. [Longformer](https://github.com/allenai/longformer)

---

## Setup

For this project you need:
- Python 3.6.2
- Numpy
- TensorFlow 1.4
- NLTK version 3.2.4 (or 3.2.5)
- Torch
- Transformer
- Spacy
- Pandas
- SpellChecker

## Database

This project has two folders for datasets:
- [data](https://github.com/AshutoshJani/exemplary/tree/main/data): This is data for the Longformer and some example data has been provided.
- [Data](https://github.com/AshutoshJani/exemplary/tree/main/Data): This is conversational data for the ChatLearner and links for the dataset can be found in a readme in the folder

## Run Project

This project has a chat UI and a development backend that can be accessed by running different files

To run the project **as a bootstrap webapp**, run the following command from the flask directory(exemplary/flask):
`python app.py`

To run the project **in development mode, on the terminal**, run the following command from the flask directory(exemplary/flask):
`python app2.py`

---
