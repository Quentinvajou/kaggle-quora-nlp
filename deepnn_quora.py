# coding: utf-8

###################################
###################################
# Introduction
#
# *[Inspired by TensorFlow tutorial](https://www.tensorflow.org/versions/r0.12/tutorials/word2vec/)*
#
# *by Quentin Vajou*
#
# *May 2017*
###################################
###################################

import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns

# import string
import re

import tensorflow as tf
import time

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
from gensim.models import word2vec
import gensim

# os.chdir("./Quora NLP")
# os.getcwd()
# print(os.path.dirname(sys.argv[0]))

pd.options.display.max_colwidth = 200

sess = tf.InteractiveSession()

pal = sns.color_palette()
global t

print("File size :")
for f in os.listdir('./input'):
     print(f + '   ' + str(round(os.path.getsize('./input/' + f)/1000000, 2)) + 'MB')

df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')


# %% function definition

def clean_data(df_train):
    df_train_c = pd.DataFrame()
    df_train.question1 = df_train.question1.astype(str)
    df_train.question2 = df_train.question2.astype(str)

    df_train_c["question1"] = df_train.question1.map(lambda x: re.sub(r'\W+', ' ', x))
    df_train_c["question2"] = df_train.question2.map(lambda y: re.sub(r'\W+', ' ', y))

    # df_train_c["question1"] = df_train.question1.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))

    df_train_c["question1"] = df_train_c.question1.str.lower()
    df_train_c["question2"] = df_train_c.question2.str.lower()

    return df_train_c

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and pooling.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def deepnn(x):
    # 1st convo layer
    W_conv1 = weight_variable([50, 1, 1, 32])
    b_conv1 = bias_variable([32])

    x_txt = tf.reshape(x, [-1, 300, 300, 1])

    h_conv1 = tf.nn.relu(conv2d(x_txt, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 2nd convo layer
    W_conv2 = weight_variable([50, 1, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable([125 * 125 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 125 * 125 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout to reduce overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer
    W_fc2 = weight_variable([1024, 1])
    b_fc2 = bias_variable([1])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob

def next_batch(num, data):
    """
    Return a total of `num` samples from the array `data`.
    """
    idx = np.arange(0, len(data))  # get all possible indexes
    np.random.shuffle(idx)  # shuffle indexes
    idx = idx[0:num]  # use only `num` random indexes
    data_shuffle = [data.ix[i] for i in idx]  # get list of `num` random samples
    data_shuffle = pd.DataFrame(data_shuffle)  # get back pandas array

    return data_shuffle

def word2vec_train(df_train):

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    sentence_cloud = pd.Series(df_train.question1.tolist() + df_train.question2.tolist() + df_test.question1.tolist() + df_test.question2.tolist()).astype(str)
    # sentence_cloud_r = next_batch(1000, sentence_cloud)

    # part 1 : train part of the whole data
    # sentence_cloud_r = sentence_cloud_r[0].str.lower()
    # sentence_cloud_r = sentence_cloud_r.str.split()

    # part 2: train all words
    sentence_cloud = sentence_cloud.str.lower().str.split()
    # print(type(sentence_cloud))
    # print(sentence_cloud[0])
    print("training model...")
    model = word2vec.Word2Vec(sentence_cloud, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
    model.init_sims(replace=True)

    model_name = "word_quora_model_clean"
    model.save(model_name)

    return model

def vec2sentence(row, model):
    row_vec = []
    for word in row:
        if word in model.wv.vocab:
            row_vec.append(model[word])
    return row_vec

def vectorize_sentence(model, df_train_c):
    question = pd.Series(df_train_c.question1 + " " + df_train_c.question2)
    question = question.str.split()
    question = question.apply(lambda row: vec2sentence(row, model))

    return question


def main(_):
    df_train_c = clean_data(df_train)
    # model = word2vec_train(df_train_c)

    # model = gensim.models.Word2Vec.load("first_word_model")
    # model = gensim.models.Word2Vec.load_word2vec_format('freebase-vectors-skipgram1000/knowledge-vectors-skipgram1000.bin', binary=True)
    # model = gensim.models.Word2Vec.load("word_quora_model")
    model = gensim.models.Word2Vec.load("word_quora_model_clean")

    # model.doesnt_match("brazil france germany spain".split())
    # model.doesnt_match("facebook google amazon apple microsoft costco".split())
    # model.most_similar("most")
    # print(model["dns"])

    nn_question = vectorize_sentence(model, df_train_c)
    # print(df_train_c.ix[404287])
    # print(nn_question.ix[404287])
    nn_question = pd.concat([nn_question, df_train.is_duplicate], axis=1, keys=['vec_sentence', 'is_duplicate'])
    # print(np.shape(nn_question.ix[1][0]))
    # print(nn_question.vec_sentence.map(lambda x: len(x)).max())

    batch_train = next_batch(1, nn_question)
    print(batch_train.vec_sentence.iloc[0])


    x = tf.placeholder(tf.float32, [None, None, None])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # Build graph for deep network
    y_conv, keep_prob = deepnn(x)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    # Training
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Model eval
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())


    for i in range(1000):
        t = time.time()
        batch_train = next_batch(50, nn_question)
        train_step.run(feed_dict={x: batch_train.vec_sentence, y_: batch_train.is_duplicate, keep_prob: 1.0})
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_train.vec_sentence, y_: batch_train.is_duplicate, keep_prob: 1.0})
            print("step : %d, training accuracy: %2f (%3f sec)"%(i, train_accuracy, time.time() - t))

    # Model evaluation
    #print(accuracy.eval(feed_dict={x: batch_test.text}))



# %% launch main
if __name__ == '__main__':
    tf.app.run(main=main)
