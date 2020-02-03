# Hatespeech-Classification-CNN
Performing hatespeech classification with a Convulotional Neural Network model. Comparing with logistic regression and simple version of XGBoost.

In order to run the project with the premade vectors from GloVe, you need to do the following:

1. Download the file containing the pretrained tweet vectors from https://nlp.stanford.edu/projects/glove/
2. Unzip the file and convert the files containing the keyed vector to a correct format using the following terminal commands:
``python -m gensim.scripts.glove2word2vec --input [filename25d] --output glove-twitter-25.txt``
``python -m gensim.scripts.glove2word2vec --input [filename50d] --output glove-twitter-50.txt``
``python -m gensim.scripts.glove2word2vec --input [filename100d] --output glove-twitter-100.txt``
``python -m gensim.scripts.glove2word2vec --input [filename200d] --output glove-twitter-200.txt``
3. Place the files you have retrieved in the folder "glove.twitter.27B". You should be good to go.

The file "data_merged.csv" contains all the data, which is preprocessed and partitioned when a model file is run. 

To run a model, simply type "python [modelfilename.py]" (example: python cnn.py). Adding arguments are possible. 

The following arguments exist.

For cnn.py: 

    - -regval : the regularizing l2-value. 
    - -activationfunction: which activation function to use in the intermediate neurones. Suggestions can be "relu" or "tanh"
    - -use_pretrained_vecs: True or False. If true, the one uses the glove vectors for the training. Default is False.
    - -emb_dim: default is 200, but can also (when using pretrained vectors) be 25, 50 and 100. Can be any number of dimensions when not using pretrained vectors. 
   
For xgboost_1.py:
   
    - -seed: the random state seed for partitioning the data. Is an integer. Default 123
    - -use_pretrained_vecs: True or False. If true, the one uses the glove vectors for the training. Default is False.
    - -emb_dim: default is 200, but can also (when using pretrained vectors) be 25, 50 and 100. Can be any number of dimensions when not using pretrained vectors. 
    
