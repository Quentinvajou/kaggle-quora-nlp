
# coding: utf-8

###################################
###################################
# Introduction
#
# *[Inspired by Python notebook](https://www.kaggle.com/anokas/quora-question-pairs/data-analysis-xgboost-starter-0-35460-lb)[ by anokas](https://www.kaggle.com/anokas)*
#
# *by Quentin Vajou*
#
# *April 2017*
###################################
###################################

# %%

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

pal = sns.color_palette()
%matplotlib inline

# Input data files are available in the "./input/" directory.

print("File size :")
for f in os.listdir('./input'):
     print(f + '   ' + str(round(os.path.getsize('./input/' + f)/1000000, 2)) + 'MB')

# %%

df_train = pd.read_csv('./input/train.csv')
df_train.head()


# - id : looks like the row ID
# - qid{1, 2} : unique ID of each question in the pair
# - question{1, 2} : the text content of the question
# - is_duplicate : what we're trying to predict : whether the pair of question is a duplicate or not

# %%

print('Nb of question pairs in training set : {}' .format(len(df_train)))
print('Duplicate pairs : {}%' .format(round(df_train['is_duplicate'].mean()*100, 2)))
qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
print('total number of questions in the training data : {}'.format(len(np.unique(qids))))
print('Number of questions that appear multiple times : {}'.format(np.sum(qids.value_counts()>1)))

plt.figure(figsize=(12, 5))
plt.hist(qids.value_counts(), bins=40)
plt.yscale('log', nonposy='clip')
plt.title('Histogram of question appearance count on a log scale')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
plt.show()


# ## Test Submission

# %%
# TEST SUBMISSION

from sklearn.metrics import log_loss

p = df_train["is_duplicate"].mean()
#print(np.zeros_like(df_train["is_duplicate"].head(15)) + p)
#print(df_train["is_duplicate"])
print("Predicted Score : ", log_loss(df_train["is_duplicate"], np.zeros_like(df_train["is_duplicate"]) + p))

df_test = pd.read_csv('./input/test.csv')
#print(df_test.head())
sub = pd.DataFrame({'test_id': df_test["test_id"], 'is_duplicate': p})
sub.to_csv("naive_submission.csv", index=False)
sub.head()


# 0.55 on the public leaderboard. The discrepency between the leaderboard and the local score (~0.658) indicates that the distribution of value on the leaderboard would lead to problems on the validation set.

# %%
# DIGGING TEST SET

print(df_test.head())


# %%

print("Total number of questions pairs in testing set : {}".format(len(df_test)))

# 2.3 millions questions. As explained in the Data section of the competition theres's a lot of auto-generated data (deter hand-labelling).

# %%
# TEXT ANALYSIS

# Let's take a look at what's inside the data. Looking through the train set as the test set contains auto-genrated data.

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
#print(train_qs.head(15))

dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)

plt.figure(figsize=(10,3))
plt.hist(dist_train, bins=200, normed=True, color=pal[2], range=[0,200], label='train')
plt.hist(dist_test, bins=200, normed=True, color=pal[1], range=[0,200], label='test', alpha=0.5)
plt.title('histogram of the repartition of the questions in length')
plt.xlabel('number of characters in question')
plt.ylabel('Probability')
plt.show()

print("mean-train: {}\nstd-train:{}\nmax-train: {}\n\nmean-test: {}\nstd-test : {}\nmax-test: {}".format(dist_train.mean(), dist_train.std(), dist_train.max(), dist_test.mean(), dist_test.std(), dist_test.max()))


# Most of the questions have between 15 and 150 characters in them. We can notice a fall at 150 characters on the train set (Quora's limit?). The graph has been cut at 200 for readability purposes. Only a few outliers outside that area.

# %%

dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(10,3))
plt.hist(dist_train, bins=50, normed=True, color=pal[2], range=[0,100], label='train')
plt.hist(dist_test, bins=50, normed=True, color=pal[1], range=[0,100], label='test', alpha=0.5)
plt.title("Repartition of the number of words in questions")
plt.legend()
plt.xlabel("Number of words")
plt.ylabel("Probability")
plt.show()


# It looks like the train set is more dense in the number of questions with around mean number of words. Whereas test set seems more spread.

# %%

from wordcloud import WordCloud

cloud = WordCloud(width=1440, height=1080).generate("".join(train_qs.astype(str)))
plt.figure(figsize=(10,10))
plt.imshow(cloud)
plt.axis('off')





# %%
# SEMANTIC ANALYSIS

qmarks = np.mean(train_qs.apply(lambda x: '?' in x))
math_signs = np.mean(train_qs.apply(lambda x: '[math]' in x))
fullstop = np.mean(train_qs.apply(lambda x: '.' in x))
capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))
capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))
numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))

print('? => {:.1f}%'.format(qmarks*100))
print('[math] => {:.1f}%'.format(math_signs*100))
print('. => {:.1f}%'.format(fullstop*100))
print('capital first => {:.1f}%'.format(capital_first*100))
print('capitals => {:.2f}%'.format(capitals*100))
print('numbers => {:.1f}%'.format(numbers*100))



# %%
# INITIAL FEATURE ANALYSIS

#Here we'll try to make a feature by calculating the percentage of common words between 2 questions.

from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def match_words_share (row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().replace('?', '').split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().replace('?', '').split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # only stopwrods generated questions
        return 0
    q1words_match = [w for w in q1words.keys() if w in q2words]
    q1words_match = [w for w in q2words.keys() if w in q1words]
    R = (len(q1words_match) + len(q1words_match)) / (len(q1words) + len(q2words))

    return R


# %%

train_word_match = df_train.apply(match_words_share, axis=1, raw=True)


# %%

plt.figure(figsize=(10,3))
plt.hist(train_word_match[df_train['is_duplicate'] == 1], bins=20, normed=True, label='is duplicate')
plt.hist(train_word_match[df_train['is_duplicate'] == 0], bins=20, normed=True, label='is not duplicate', alpha=0.5)
plt.title("Repartition of percentage of words in common (out of stopwords)")
plt.xlabel("word match share")
plt.legend()
plt.show()


# %%
# DATA PREPARATION

x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['word_match'] = train_word_match
x_test['word_match'] = df_test.apply(match_words_share, axis=1, raw=True)

y_train = df_train['is_duplicate']
#print(y_train)


# %%
# RE-BALANCING THE DATA
# As described in the notebook from anokas, there seems to be around 16.5% postive class in the test set against 37% in our training set (the link is broken so I don't know how that calculation is done).
# So we want to balance the result so as to train our model with the same proportion of positive classes.

pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]

p = 0.165
scale = len(pos_train) / (len(pos_train) + len(neg_train))
print(scale)

neg_train = pd.concat([neg_train, neg_train])

scale = len(pos_train) / (len(pos_train) + len(neg_train))
print(scale)

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

print(len(x_train), len(y_train))

# %%
# SPLIT DATA

from sklearn.cross_validation import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

# %%
# XGBOOST

import xgboost as xgb

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'd_train'), (d_valid, 'd_valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

# %%
# RESULTS

d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb.csv', index=False)
