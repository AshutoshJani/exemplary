# Here the dataset for the ChatLearner should go. In our requirement we used the Papaya Conversational Dataset

## Papaya Conversational Data Set
Papaya Data Set is the best (cleanest and well-organized) free English conversational data you can find on the web for training a chatbot. Here are some details:

1. The data are composed of two sets: the first set was handcrafted, and we created the samples in order to maintain a consistent role of the chatbot, who can therefore be trained to be polite, patient, humorous, philosophical, and aware that he is a robot, but pretend to be a 9-year old boy named Papaya; the second set was cleaned from some online resources, including the scenario conversations designed for training robots, the Cornell movie dialogs, and cleaned Reddit data.

2. The training data set is split into three categories: two subsets will be augmented/repeated during the training, with different levels or times, while the third will not. The augmented subsets are to train the model with rules to follow, and some knowledge and common senses, while the third subset is just to help to train the language model.

3. The scenario conversations were extracted and reorganized from http://www.eslfast.com/robot/. If your model can support context, it would work much better by utilizing these conversations.

4. The original Cornell data set can be found at [here](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). We cleaned it using a Python script (the script can also be found in the Corpus folder); we then cleaned it manually by quickly searching certain patterns. 

5. For the Reddit data, a cleaned subset (about 110K pairs) is included in this repository. The vocab file and model parameters are created and adjusted based on all the included data files. In case you need a larger set, you can also find scripts to parse and clean the Reddit comments in the Corpus/RedditData folder. In order to use those scripts, you need to download a torrent of Reddit comments from a torrent link [here](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/). 
Normally a single month of comments is big enough (can generated 3M pairs of training samples roughly). You can tune the parameters in the scripts based on your needs. 

6. The data files in this data set were already preprocessed with NLTK tokenizer so that they are ready to feed into the model using new tf.data API in TensorFlow.


## This description has been taken from [this repo](https://github.com/bshao001/ChatLearner) by Bo Shao
