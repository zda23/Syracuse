#!/usr/bin/env python
# coding: utf-8

# In[155]:


import nltk
from nltk.corpus import sentence_polarity
import random
from nltk.corpus import brown
import re


# In[159]:


sentences = sentence_polarity.sents()
regex = r"[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]"
substr = ""
documents_pos = [(sent) for sent in sentence_polarity.sents(categories='pos')]
documents_neg = [(sent) for sent in sentence_polarity.sents(categories='neg')]
docs_pos = ' '.join(str(e) for e in documents_pos)
cleaned_positives = re.sub(regex, substr, docs_pos, 0, re.MULTILINE)
positives = nltk.word_tokenize(cleaned_positives)
docs_neg = ' '.join(str(e) for e in documents_neg)
cleaned_negatives = re.sub(regex, substr, docs_neg, 0, re.MULTILINE)
negatives = nltk.word_tokenize(cleaned_negatives)
documents = [(sent, cat) for cat in sentence_polarity.categories()
    for sent in sentence_polarity.sents(categories=cat)]
tagged_pos = nltk.pos_tag(positives)
tagged_neg = nltk.pos_tag(negatives)


# In[203]:


def Convert(tup, dic):
    dic = dict(tup)
    return dic
     
tups = tagged_pos
dictionary = {}
tups_neg = tagged_neg
dictionary_neg = {}
positive_dict = Convert(tups, dictionary)
negative_dict = Convert(tups_neg, dictionary_neg)
pos_adj = [k for k, v in positive_dict.items() if v == 'JJ']
print("TOP 50 ADJECTIVES IN POSITIVE SENTENCES ARE: \n"  + str(pos_adj[:50]))
pos_adv = [k for k, v in positive_dict.items() if v == 'RB']
print("TOP 50 ADVERBS IN POSITIVE SENTENCES ARE: \n"  + str(pos_adv[:50]))
pos_verbs = [k for k, v in positive_dict.items() if v == 'VB']
print("TOP 50 VERBS IN POSITIVE SENTENCES ARE: \n"  + str(pos_verbs[:50]))
neg_adj = [k for k, v in negative_dict.items() if v == 'JJ']
print("TOP 50 ADJECTIVES IN NEGATIVE SENTENCES ARE: \n"  + str(neg_adj[:50]))
neg_adv = [k for k, v in negative_dict.items() if v == 'RB']
print("TOP 50 ADVERBS IN NEGATIVE SENTENCES ARE: \n"  + str(neg_adv[:50]))
neg_verbs = [k for k, v in negative_dict.items() if v == 'VB']
print("TOP 50 VERBS IN NEGATIVE SENTENCES ARE: \n"  + str(neg_verbs[:50]))


# In[204]:


neg_words_list = [word for (sent) in tagged_neg for word in sent]
neg_words = nltk.FreqDist(neg_words_list)
neg_word_items = neg_words.most_common(50)
print(neg_word_items)


# In[205]:


pos_words_list = [word for (sent) in tagged_pos for word in sent]
pos_words = nltk.FreqDist(pos_words_list)
pos_word_items = pos_words.most_common(50)
print(pos_word_items)


# In[4]:


random.shuffle(documents)
all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(2000)
word_features = [word for (word,count) in word_items]


# In[5]:


def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features


# In[6]:


featuresets = [(document_features(d, word_features), c) for (d, c) in documents]


# In[41]:


train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[8]:


nltk.classify.accuracy(classifier, test_set)


# In[9]:


negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing',
'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither',
'nor']


# In[11]:


def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
# go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or
(word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in
word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    return features


# In[12]:


NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in documents]


# In[42]:


NOT_train_set, NOT_test_set = NOT_featuresets[1000:], NOT_featuresets[:1000]
NOT_classifier = nltk.NaiveBayesClassifier.train(NOT_train_set)
nltk.classify.accuracy(NOT_classifier, NOT_test_set)


# In[14]:


stopwords = nltk.corpus.stopwords.words('english')


# In[15]:


negationwords.extend(['ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn',
'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan',
'shouldn', 'wasn', 'weren', 'won', 'wouldn'])
newstopwords = [word for word in stopwords if word not in negationwords]


# In[16]:


new_all_words_list = [word for (sent,cat) in documents for word in sent if word not in newstopwords]
new_all_words = nltk.FreqDist(new_all_words_list)
new_word_items = new_all_words.most_common(2000)
new_word_features = [word for (word,count) in new_word_items]


# In[17]:


def new_NOT_features(document, new_word_features, negationwords):
    features = {}
    for word in new_word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
# go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or
(word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in
new_word_features)
        else:
            features['V_{}'.format(word)] = (word in new_word_features)
    return features


# In[18]:


new_NOT_featuresets = [(new_NOT_features(d, new_word_features, negationwords), c) for (d, c) in documents]


# In[43]:


new_NOT_train_set, new_NOT_test_set = new_NOT_featuresets[1000:], new_NOT_featuresets[:1000]
new_NOT_classifier = nltk.NaiveBayesClassifier.train(new_NOT_train_set)
nltk.classify.accuracy(new_NOT_classifier, new_NOT_test_set)


# In[21]:


from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()


# In[22]:


finder = BigramCollocationFinder.from_words(all_words_list)
bigram_features = finder.nbest(bigram_measures.chi_sq, 500)


# In[23]:


def bigram_document_features(document, word_features, bigram_features):
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    for bigram in bigram_features:
        features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)
    return features


# In[24]:


bigram_featuresets = [(bigram_document_features(d, word_features, bigram_features), c) for (d, c) in documents]


# In[25]:


new_bigram_featuresets = [(bigram_document_features(d, new_word_features, bigram_features), c) for (d, c) in documents]


# In[53]:


bigram_train_set, bigram_test_set = bigram_featuresets[1000:], bigram_featuresets[:1000]
bigram_classifier = nltk.NaiveBayesClassifier.train(bigram_train_set)
nltk.classify.accuracy(bigram_classifier, bigram_test_set)


# In[54]:


new_bigram_train_set, new_bigram_test_set = new_bigram_featuresets[1000:], new_bigram_featuresets[:1000]
new_bigram_classifier = nltk.NaiveBayesClassifier.train(new_bigram_train_set)
nltk.classify.accuracy(new_bigram_classifier, new_bigram_test_set)


# In[28]:


def POS_features(document, word_features):
    document_words = set(document)
    tagged_words = nltk.pos_tag(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features


# In[29]:


POS_featuresets = [(POS_features(d, word_features), c) for (d, c) in documents]
new_POS_featuresets = [(POS_features(d, new_word_features), c) for (d, c) in documents]


# In[92]:


print('num nouns', POS_featuresets[0][0]['nouns'])
print('num verbs', POS_featuresets[0][0]['verbs'])
print('num adjectives', POS_featuresets[0][0]['adjectives'])
print('num adverbs', POS_featuresets[0][0]['adverbs'])


# In[44]:


POS_train_set, POS_test_set = POS_featuresets[1000:], POS_featuresets[:1000]
POS_classifier = nltk.NaiveBayesClassifier.train(POS_train_set)
nltk.classify.accuracy(POS_classifier, POS_test_set)


# In[45]:


new_POS_train_set, new_POS_test_set = new_POS_featuresets[1000:], new_POS_featuresets[:1000]
new_POS_classifier = nltk.NaiveBayesClassifier.train(new_POS_train_set)
nltk.classify.accuracy(new_POS_classifier, new_POS_test_set)


# In[39]:


def cross_validation_accuracy(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    accuracy_list = []
    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    # find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)


# In[40]:


num_folds = 5
cross_validation_accuracy(num_folds, featuresets)
cross_validation_accuracy(num_folds, POS_featuresets)
cross_validation_accuracy(num_folds, NOT_featuresets)
cross_validation_accuracy(num_folds, new_NOT_featuresets)


# In[46]:


cross_validation_accuracy(num_folds, new_POS_featuresets)
cross_validation_accuracy(num_folds, bigram_featuresets)
cross_validation_accuracy(num_folds, new_bigram_featuresets)


# In[47]:


goldlist = []
predictedlist = []
for (features, label) in test_set:
    goldlist.append(label)
    predictedlist.append(classifier.classify(features))


# In[48]:


POS_goldlist = []
POS_predictedlist = []
for (features, label) in POS_test_set:
    POS_goldlist.append(label)
    POS_predictedlist.append(POS_classifier.classify(features))


# In[49]:


new_POS_goldlist = []
new_POS_predictedlist = []
for (features, label) in new_POS_test_set:
    new_POS_goldlist.append(label)
    new_POS_predictedlist.append(new_POS_classifier.classify(features))


# In[50]:


NOT_goldlist = []
NOT_predictedlist = []
for (features, label) in NOT_test_set:
    NOT_goldlist.append(label)
    NOT_predictedlist.append(NOT_classifier.classify(features))


# In[51]:


new_NOT_goldlist = []
new_NOT_predictedlist = []
for (features, label) in new_NOT_test_set:
    new_NOT_goldlist.append(label)
    new_NOT_predictedlist.append(new_NOT_classifier.classify(features))


# In[55]:


bigram_goldlist = []
bigram_predictedlist = []
for (features, label) in bigram_test_set:
    bigram_goldlist.append(label)
    bigram_predictedlist.append(bigram_classifier.classify(features))


# In[56]:


new_bigram_goldlist = []
new_bigram_predictedlist = []
for (features, label) in new_bigram_test_set:
    new_bigram_goldlist.append(label)
    new_bigram_predictedlist.append(new_bigram_classifier.classify(features))


# In[57]:


cm = nltk.ConfusionMatrix(goldlist, predictedlist)
print(cm.pretty_format(sort_by_count=True, truncate=9))


# In[58]:


POS_cm = nltk.ConfusionMatrix(POS_goldlist, POS_predictedlist)
print(POS_cm.pretty_format(sort_by_count=True, truncate=9))


# In[60]:


new_POS_cm = nltk.ConfusionMatrix(new_POS_goldlist, new_POS_predictedlist)
print(new_POS_cm.pretty_format(sort_by_count=True, truncate=9))


# In[61]:


NOT_cm = nltk.ConfusionMatrix(NOT_goldlist, NOT_predictedlist)
print(NOT_cm.pretty_format(sort_by_count=True, truncate=9))


# In[62]:


new_NOT_cm = nltk.ConfusionMatrix(new_NOT_goldlist, new_NOT_predictedlist)
print(new_NOT_cm.pretty_format(sort_by_count=True, truncate=9))


# In[63]:


bigram_cm = nltk.ConfusionMatrix(bigram_goldlist, bigram_predictedlist)
print(bigram_cm.pretty_format(sort_by_count=True, truncate=9))


# In[64]:


new_bigram_cm = nltk.ConfusionMatrix(new_bigram_goldlist, new_bigram_predictedlist)
print(new_bigram_cm.pretty_format(sort_by_count=True, truncate=9))


# In[65]:


def eval_measures(gold, predicted):
# get a list of labels
    labels = list(set(gold))
# these lists have values for each label
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
# for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab: TP += 1
            if val == lab and predicted[i] != lab: FN += 1
            if val != lab and predicted[i] == lab: FP += 1
            if val != lab and predicted[i] != lab: TN += 1
# use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))
    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]),             "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))


# In[66]:


eval_measures(goldlist, predictedlist)


# In[67]:


eval_measures(POS_goldlist, POS_predictedlist)


# In[68]:


eval_measures(new_POS_goldlist, new_POS_predictedlist)


# In[69]:


eval_measures(NOT_goldlist, NOT_predictedlist)


# In[70]:


eval_measures(new_NOT_goldlist, new_NOT_predictedlist)


# In[71]:


eval_measures(bigram_goldlist, bigram_predictedlist)


# In[72]:


eval_measures(new_bigram_goldlist, new_bigram_predictedlist)

