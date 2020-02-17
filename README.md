# SequenceFramework
Process &amp; Select Important Sequence.

Select important text/words in sentence sequences(documents).

By proposing a new NLP sequence processing deep learning framework.

(Implement by TF2.0)

Models/data: https://drive.google.com/drive/u/1/folders/1DOpFaLUaYyzTBgLqVcjgUwfI33m5frQK
* one-hot encoding training.csv (A tri-gram is represented by 1377-dim vectors. Each profile has 182 tri-grams, including padding. Last column is the label.)
* one-hot encoding testing.csv

## 20 newsgroups Dataset
* The 20 newsgroups text dataset: http://qwone.com/~jason/20Newsgroups/
* 20news-bydate.tar.gz - 20 Newsgroups sorted by date; duplicates and some headers removed (18846 documents)
### Reuters dataset (baseline model)
* Reuters dataset in Keras https://keras.io/datasets/ 、 https://keras.io/examples/reuters_mlp/ 、 https://github.com/nltk/nltk_data/tree/gh-pages/packages/corpora

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
   * pick max length = oulier = 360

* all words #: 30979
    * words frequency diagram in results/: words_dist_diagram.xlsx
    * pick 8352 words (count>=10)


* 80% for training: 8260 train sequences
* 20% for testing: 2066 test sequences

### Preprocessing REF
* https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

## Framework
* MTL(2 optimizers)?weighted loss(1 optmizer)?
* end2end
### Hyper-parameters & Experiments
* max_words、max_length
* hidden_dim
* init_w、init_b
* activation function slope: portion
* model1 arch.、model2 arch.
* batch_size
* loss seq: seq_num
* optimizer
* weighted loss: alpha、beta、gamma
* input weight or binary: pred_imp or pred_imp2
* Experiments parameters
detail link: https://docs.google.com/spreadsheets/d/1iSLUaFhr27HRi2YB8bPCeuOCmBx18nkiCAKWk4Fg8JA/edit#gid=0
### model1: Embedder + (Encoder) + Filter
* Input: ID of words (int)
* Output: embedding vectors(float32, tensors) & **filter**(float32, tensors)
    * round filter to binary number (0 or 1) => **filter2**
* loss1(alpha): mean of filter
* loss3(gamma): consecutive of filter2
* ones_num: numbers of 1's portion
### model2: Classifier
* Input: embedding vectors & filter2 
* Output: **classID**
* loss2(beta): categorical cross entropy of classID
* acc_rate: accuracy in classification

## Functions, Records & Statistics
### losses_metrics.xlsx
* Train / Test family classify loss of each epoch
* Train / Test filter loss of each epoch、可調整希望模型1越多或是0越多、可調整已weight輸入或以binary輸入
* Train / Test sequence loss of each epoch、可調整連續之windows size (越低代表越連續)
* Train / Test accuracy of each epoch
* Train / Test 1's number of each epoch
* Train / Test each class' average intra-cosine-similarity of embedding vectors of each epoch
* Train / Test each class' average intra-cosine-similarity after Filter module of each epoch
* Train / Test inter-classes' cosine-similarity of embedding vectors of each epoch
* Train / Test inter-classes' cosine-simlarity after Filter module of each epoch
### caseStudy_result.xlsx
* Original input text of each testing data in each epoch
* After Filter module's text of each testing data in each epoch
* True label of each testing data in each epoch
* Predicted label of each testing data in each epoch
  
***
### Below Deprecated
Possible solutions:
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
