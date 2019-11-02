#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys,tqdm
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.text import *

from collections import Counter
import pandas as pd
import shutil
import pickle
import gc
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[2]:


def basic_statistics(all_length):
    '''
    input: length list of elements e.g.[1,1,1,3,5,9,4,2,1,3,54,78,5...]
    output1: mean、std、mode、min、q1、median(q2)、q3、max、iqr、outlier、far out
    output2: statistics graph、10%~90% form
    '''
    stat_dict = {}
    stat_dict['mean'] = np.mean(all_length)
    stat_dict['std'] = np.std(all_length)
    stat_dict['mode'] = np.argmax(np.bincount(all_length))
    stat_dict['min'] = np.min(all_length)
    stat_dict['q1'] = np.quantile(all_length,0.25)
    stat_dict['median'] = np.quantile(all_length,0.5)
    stat_dict['q3'] = np.quantile(all_length,0.75)
    stat_dict['max'] = np.max(all_length)
    stat_dict['iqr'] = stat_dict['q3'] - stat_dict['q1']
    stat_dict['outlier'] = stat_dict['q3'] + 1.5*stat_dict['iqr']
    stat_dict['far_out'] = stat_dict['q3'] + 3*stat_dict['iqr']
    for i in [10,20,30,40,50,60,70,80,90,100]:
        stat_dict[str(i)+'%'] = np.percentile(all_length,i)
    return pd.DataFrame.from_dict(stat_dict,orient='index',columns=['length'])


# In[3]:


max_words = 8352#8352 #Top most frequent words to consider. Any less frequent word will appear as oov_char value in the sequence data.
max_length = 360#360


# In[4]:


word_index = reuters.get_word_index()
print('all_words#:',len(word_index))
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,maxlen=max_length,
                                                         test_split=0.2,seed=830913)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


# In[13]:


# train_len = [len(x) for x in x_train]
# test_len = [len(x) for x in x_test]
# all_len = train_len
# all_len.extend(test_len)
# basic_statistics(all_len)


# In[15]:


# df = pd.DataFrame(all_len)
# df.to_excel('./results/length_dist.xlsx', header=False, index=False)
# df


# In[21]:


# train_words = []
# for x in x_train:
#     train_words.extend(x)
# test_words = []
# for x in x_test:
#     test_words.extend(x)
# all_words = train_words
# all_words.extend(test_words)
# all_statistcs = Counter(all_words)
# all_statistcs


# In[28]:


# df = pd.DataFrame.from_dict(dict(all_statistcs), orient = 'index')
# df.to_excel('./results/words_dist2.xlsx', header=False, index=True)
# df


# In[5]:


trainX = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_length,padding='post',value=0)
testX = tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_length,padding='post',value=0)
print(trainX.shape, testX.shape)


# In[14]:


hidden_dim = 256
do = 0.1


# ## Graph execution
# ### Embedder

# In[42]:


int_id = Input(shape=(max_length,), dtype='int32', name='int_ids') # 輸入的api funvtion name ID
int_ids = Masking(mask_value=0)(int_id)
sent_emb = Embedding(max_words, hidden_dim,input_length=max_length
                    ,trainable=True,name='glove_emb')(int_ids) 


# ### Encoder

# In[43]:


rnn = GRU(int(hidden_dim/2),return_sequences=True,return_state=False,name='common_extract'
                      ,trainable=True)(sent_emb)
rnn = BatchNormalization(name='bn')(rnn)


# ### Filter

# In[44]:


fil = TimeDistributed(Dense(1,activation='sigmoid',
                             name='filter_out'),name='TD2')(rnn)


# ### Classfier

# In[47]:


mul = Multiply()([fil,sent_emb])
clf = LSTM(int(hidden_dim/2),dropout=do,recurrent_dropout=do,name='lstm')(mul)
clf = BatchNormalization(name='bn3')(clf)
clf = Dense(max(y_train)+1,activation='softmax',name='clf')(clf)


# ## Compile

# In[48]:


model = Model(inputs=int_id, outputs = clf)
model.summary()


# In[53]:


# loss
import keras.backend as K
def custom_objective(layer):
    return K.sum(layer.output)
#     return K.sum(layer.output)
# kk = tf.keras.backend.ea
model.compile(loss=custom_objective(model.get_layer(name='TD2')),optimizer='adam')


# ## Eager Execution

# In[52]:


# whole model
do = 0
init = tensorflow.keras.initializers.Ones()
class base_model(Model):
    def __init__(self):
        super(base_model, self).__init__()
        self.mask = Masking(mask_value=0)
        self.emb = Embedding(max_words, hidden_dim,input_length=max_length
                    ,trainable=True,name='glove_emb')
        self.rnn1 = GRU(int(hidden_dim/2),return_sequences=True,return_state=False,name='common_extract'
                      ,trainable=True)
        self.bn1 = BatchNormalization(name='bn1')
        self.fil = Dense(1,activation='hard_sigmoid',kernel_initializer=init,bias_initializer=init,name='filter_out')
        #self.fil = TimeDistributed(Dense(1,activation='sigmoid', name='filter_out'),name='TD2')
        self.mul = Multiply()
        self.rnn2 = Bidirectional(GRU(int(hidden_dim/2),dropout=do,recurrent_dropout=do,name='lstm'))
        self.rnn3 = LSTM(int(hidden_dim/2))
        self.bn2 = BatchNormalization(name='bn2')
        self.out = Dense(max(y_train)+1,activation='softmax',name='clf')
    def transform(self,x):
        return tf.math.round(x)
    def call(self,x):
        x = self.mask(x)
        x1 = self.emb(x)
        x = self.rnn1(x1)
        x = self.bn1(x)
        y = self.fil(x)
        y1 = self.transform(y)
        x2 = self.mul([y1,x1])
        x = self.rnn2(x2) #x
        x = self.bn2(x)
        y2 = self.out(x)
        return y,y1,y2
        #return y,y1,y2,x2
        
model = base_model()


# In[294]:


# partial1 model
init_w = tensorflow.keras.initializers.Constant(value=0.9)
init_b = tensorflow.keras.initializers.Constant(value=0.7)
def onezero(x):
    beta = 0.6#0.6 #0.6~1
    z = tf.where(x>=1.0, x - x + 1.0, x)
    y = tf.where(z<=0.0, z - z + 0.0, beta*z)
    return y

class base_model_1(Model):
    def __init__(self):
        super(base_model_1, self).__init__()
        self.mask = Masking(mask_value=0)
        self.emb = Embedding(max_words, hidden_dim,input_length=max_length
                    ,trainable=True,name='glove_emb')
        self.rnn1 = GRU(int(hidden_dim/2),return_sequences=True,return_state=False,name='common_extract'
                      ,trainable=True)
        self.att = Attention(name='selfatt')
        self.bn1 = BatchNormalization(name='bn1')
        #self.fil = Dense(1,activation='sigmoid',name='filter_out')
        self.fil = TimeDistributed(Dense(1,activation=onezero,kernel_initializer=init_w,bias_initializer=init_b, name='filter_out'),name='TD2') #relu/linear/step function

    def call(self,x):
        x = self.mask(x)
        x1 = self.emb(x)
        x = self.att([x1,x1])
        #x = self.rnn1(x1)
        #x = self.bn1(x)
        y = self.fil(x)
        return x1,y

model1 = base_model_1()


# In[295]:


# partial2 model
class base_model_2(Model):
    def __init__(self):
        super(base_model_2, self).__init__()
        self.mul = Multiply()
        self.rnn2 = Bidirectional(GRU(int(hidden_dim/2),dropout=do,recurrent_dropout=do,name='lstm'))
        self.rnn3 = GRU(int(hidden_dim/2))
        self.bn2 = BatchNormalization(name='bn2')
        self.out = Dense(max(y_train)+1,activation='softmax',name='clf')

    def call(self,x1,y1):
        #x2 = self.mul([y1,x1])
        x = self.rnn3(y1) #x2
        x = self.bn2(x)
        y2 = self.out(x)
        return y2
    
model2 = base_model_2()


# In[162]:


# TEST
# x = tf.random.uniform((1, max_length))
# out1,out2,out3,out4 = model(x)
# out1, out4


# In[296]:


batch_size = 128 #,reshuffle_each_iteration=True
train_ds = tf.data.Dataset.from_tensor_slices((trainX,y_train)).shuffle(trainX.shape[0]).batch(batch_size)
valid_ds = tf.data.Dataset.from_tensor_slices((testX,y_test)).batch(batch_size)


# In[297]:


def loss_object1(predictions):
    mask = tf.math.logical_not(tf.math.equal(predictions, 0))
    loss_ = tf.reduce_mean(predictions)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)
def one_percentage(predictions):
    mask = tf.math.logical_not(tf.math.equal(predictions, 0))
    loss_ = tf.reduce_mean(predictions)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)
loss_object2 = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer1 = tf.keras.optimizers.Nadam()
optimizer2 = tf.keras.optimizers.RMSprop()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_ones = tf.keras.metrics.Mean(name='train_ones')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_ones = tf.keras.metrics.Mean(name='test_ones')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# In[298]:


# seperate partial model
alpha = 0.0 #pahse1: -0.1 / 0.0 ; phase2: 0.01~0.05~0.1
@tf.function
def train_step(x,yc):
    with tf.GradientTape(persistent=False) as tape:
        emb, pred_imp = model1(x)
        #loss1 = alpha*loss_object1(pred_imp) #phase1
        #pred_imp2 = tf.math.round(pred_imp)
        #pred_imp3 = tf.clip_by_value(pred_imp,clip_value_max=1,clip_value_min=0)
        pred_imp2 = tf.math.round(pred_imp)
        loss1 = alpha*(1-loss_object1(pred_imp)) #pahse2:alpha*loss_object1(pred_imp) ; phase1: alpha*(1-loss_object1(pred_imp))
        pred_cat = model2(emb,pred_imp2) #pahse1: pred_imp; phase2; pred_imp2
        loss2 = loss_object2(yc, pred_cat)
        loss = loss1 + loss2
    trainable_variable = model1.trainable_variables
    trainable_variable.extend(model2.trainable_variables)
    gradients = tape.gradient(loss,trainable_variable)
    optimizer1.apply_gradients(zip(gradients,trainable_variable))
    
    train_loss(loss)
    train_accuracy(yc, pred_cat)
    ones = one_percentage(pred_imp)
    train_ones(ones)
    
@tf.function
def test_step(x,yc):
    emb, pred_imp = model1(x)
    #loss1 = alpha*loss_object1(pred_imp) #phase1
    #pred_imp2 = tf.math.round(pred_imp)
    #pred_imp3 = tf.clip_by_value(pred_imp,clip_value_max=1,clip_value_min=0)
    pred_imp2 = tf.math.round(pred_imp)
    loss1 = alpha*(1-loss_object1(pred_imp)) #phase2
    pred_cat = model2(emb,pred_imp2) #phase1: pred_imp ; phase2:pred_imp2
    loss2 = loss_object2(yc, pred_cat)
    #t_loss = loss1 + loss2
    t_loss = loss1 + loss2
    
    test_loss(t_loss)
    test_accuracy(yc, pred_cat)
    t_ones = one_percentage(pred_imp)
    test_ones(t_ones)


# In[55]:


#AIO
alpha = 0.1
@tf.function
def train_step(x,yc):
    with tf.GradientTape(persistent=True) as tape: #persistent=True
        pred_imp,pred_round , pred_cat = model(x)
#         pred_cat = model(x)
#         loss = alpha*loss_object1(pred_imp) + loss_object2(yc,pred_cat)
        loss1 = alpha*loss_object1(pred_imp)
        loss2 = loss_object2(yc,pred_cat)
        loss = loss_object2(yc,pred_cat)
#     gradients = tape.gradient(loss, model.trainable_variables)
    grad1 = tape.gradient(loss1, model.trainable_variables)
    grad2 = tape.gradient(loss2, model.trainable_variables)
#     optimizer1.apply_gradients(zip(gradients, model.trainable_variables))
    optimizer1.apply_gradients(zip(grad1, model.trainable_variables))
    optimizer2.apply_gradients(zip(grad2, model.trainable_variables))
#     with tf.GradientTape() as tape:
#         pred_imp , pred_cat = model(x)
#         loss2 = loss_object2(yc,pred_cat)
#         loss = alpha*loss_object1(pred_imp) + loss_object2(yc,pred_cat)
#     grad2 = tape.gradient(loss2, model.trainable_variables)
#     optimizer2.apply_gradients(zip(grad2, model.trainable_variables))

    train_loss(loss)
    train_accuracy(yc, pred_cat)
    ones = one_percentage(pred_round)
    train_ones(ones)
    
@tf.function
def test_step(x,yc):
    pred_imp,pred_round, pred_cat = model(x)
#     pred_cat = model(x)
    t_loss = alpha*loss_object1(pred_imp) + loss_object2(yc,pred_cat) 
#     t_loss = loss_object2(yc,pred_cat)
    
    test_loss(t_loss)
    test_accuracy(yc, pred_cat)
    t_ones = one_percentage(pred_round)
    test_ones(t_ones)


# In[ ]:


EPOCHS = 5000
gc.collect()
for epoch in range(EPOCHS):
    for text, labels in train_ds:
        train_step(text, labels)

    for test_text, test_labels in valid_ds:
        test_step(test_text, test_labels)

    template = 'Epoch {}, Loss: {},Ones#: {}, Accuracy: {}, Test Loss: {},Test Ones#: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_ones.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_ones.result(),
                        test_accuracy.result()*100))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_ones.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_ones.reset_states()
    test_accuracy.reset_states()


# In[ ]:





# * 不用sigmoid或是hard_sigmoid。改良relu + linear，並拆成兩個model，把round前面多加上clip
#     * sigmoid中間的變化太快(一瞬間就會掉到0或是1)，改成relu在>0~無限大(linear為了還在0~1)再去clip再round，可以看到每個epoch的變化
# * 1st phase的beta一定要>=0.6否則不會動，ones#都會是0
#     * beta=1 (放0 1進去): weight init設成1也沒用，但把bias設成1就會一開始都是ones#=1了。weight=1 bias=0.2都匯市0，bias=0.3會是0.87。0.6/0.3都是0。0.8/0.3差不多是0.5(但是train很慢acc進步很慢就是了)。0.5/0.5是從0.01開始往上升 (0 1放進去會比較難train是因為它的變化量太大，一下就是有或沒有，所以clf可能學不好，但如果是weight每次gradient進步的都是一小點就會比較容易上升)
#     * beta=0.6 (放0 1進去): 0.8/0.5都是0。0.9/0.8 從0.9一直到0。0.9/0.6差不多是從0.5但又有時候到0.7都是0.0(很難train，Nadam換個opt有時候沒用。EX變成adam 0.9/0.8才有0.96開始但如果0.9/0.75變成0開始。Rmsprop 0.9/0.8又是從0.95開始往下)。但如果都改成傳入weight就都沒問題。ones一開始大概0.5 weight平均，也不會卡住
# * 建議: 先訓練embedding weight matrix，但是要看goal是要怎樣的matrix
# * 其實他不管幾%都會train得很好，除非固定embedding，或是用更弱的clf
# * 若設定alpha，就像是regularizer term (penalty)，設越大drop越多
# * 一開始先很多ones，再越來越少個

# * 同一個opt若加入transform就會train不起來
# * 兩個不同的opt加入transform也會train不起來 (persistent、non-persis都不行)，且與BN無關
