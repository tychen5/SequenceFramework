# SequenceFramework
Process &amp; Select Important Sequence.

Select important text/words in sentence sequences(documents).

By proposing a new NLP sequence processing deep learning framework.

(Implement by TF2.0)

## 20 newsgroups Dataset
* The 20 newsgroups text dataset: http://qwone.com/~jason/20Newsgroups/
* 20news-bydate.tar.gz - 20 Newsgroups sorted by date; duplicates and some headers removed (18846 documents)
### Reuters dataset (baseline model)
* Reuters dataset in Keras https://keras.io/datasets/ 、 https://keras.io/examples/reuters_mlp/

Dataset Length Distribution:

Statistics    | Value
--------------|------------
mean	| 145.964197
std	| 145.878476
mode	| 17.000000
min	| 2.000000
q1	| 60.000000
median	| 95.000000
q3	| 180.000000
max	| 2376.000000
iqr	| 120.000000
outlier	| 360.000000
far_out	| 540.000000
10%	| 35.000000
20%	| 53.000000
30%	| 67.000000
40%	| 81.000000
50%	| 95.000000
60%	| 112.000000
70%	| 154.000000
80%	| 206.000000
90%	| 315.000000
100%	| 2376.000000

    * (diagram in results/: length_dist_diagram.xlsx)
    * pick max length = oulier (360)

* all words #: 30979
    * words frequency diagram in results/: words_dist_diagram.xlsx
    * pick 6000 words (count>=6)


* 80% for training: 8248 train sequences
* 20% for testing: 2063 test sequences

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
