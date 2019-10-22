# SequenceFramework
Process &amp; Select Important Sequence.

Select important text/words in sentence sequences(documents).

By proposing a new NLP sequence processing deep learning framework.

(Implement by TF2.0)

## Dataset
* The 20 newsgroups text dataset: http://qwone.com/~jason/20Newsgroups/
* 20news-bydate.tar.gz - 20 Newsgroups sorted by date; duplicates and some headers removed (18846 documents)

* Reuters dataset in Keras https://keras.io/datasets/ 、 https://keras.io/examples/reuters_mlp/

### Preprocessing REF
* https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

## Framework
* MTL(2 optimizers)?weighted loss(1 optmizer)?
* end2end

### Module1: Embedding
* Possible method: BERT / learnable layer...

### Module2: Filtering
* loss: sum of weights/scores
* Possible method: self-attention/LSTM/GRU
* activation function: gumble softmax/hard_sigmoid
* optimizer: Adam/Nadam/RMSprop
* Output: only binary(by rounded) / weight score



### Module3: Classifier
* input: multiply of Module2 output and Module1 output
* loss: categorical cross entropy loss
* Possible method: self-attention/LSTM/GRU/CNN
* optimizer: Adam/Nadam/RMSprop
