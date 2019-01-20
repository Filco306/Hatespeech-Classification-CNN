# Hatespeech-Classification-CNN
Performing hatespeech classification with a Convulotional Neural Network model. Comparing with logistic regression and simple version of XGBoost.

In order to run the project with the premade vectors from GloVe, you need to do the following:

1. Download the
2. Unzip the file and convert the files containing the keyed vector to a correct format using the following terminal commands:
python -m gensim.scripts.glove2word2vec --input [filename25d] --output glove-twitter-25.txt
python -m gensim.scripts.glove2word2vec --input [filename50d] --output glove-twitter-50.txt
python -m gensim.scripts.glove2word2vec --input [filename100d] --output glove-twitter-100.txt
python -m gensim.scripts.glove2word2vec --input [filename200d] --output glove-twitter-200.txt
3. Place the files you have retrieved in the folder "glove.twitter.27B". You should be good to go.
