#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os
import re
import pprint


# In[442]:


epatterns = []
epatterns.append(r'([A-Za-z.]+)@([A-Za-z.]+)\.edu')
epatterns.append(r'([A-Za-z.]+)\s@\s([A-Za-z.]+)\.edu')
epatterns.append(r'([a-z.]+)\b[(followed by &ldquo;]+.?@([a-z.]+).edu')
epatterns.append(r'([a-z.]+)\b[<del>]+.?@([a-z.]+).edu')
epatterns.append(r'(\w+)\b.[A-Z].*\b(stanford).[A-Za-z]+.edu')
epatterns.append(r'([a-z]+).at <!--.+>.(stanford).+edu')
#epatterns.append(r'([A-Za-z. ]+)\sat\s([A-Za-z.]+).edu')
#epatterns.append(r'([A-Za-z.]+)\sat\s([A-Za-z.;]+);edu')
#epatterns.append(r'([A-Za-z.-]+)@([A-Za-z.-]+)\.-e-d-u')
#epatterns.append(r'([A-Za-z.]+) @ ([A-Za-z.]+)\.edu')
#epatterns.append(r'([a-z.]+)\b[<at symbol>]+.?@([a-z.]+).edu')
#epatterns.append(r'([a-zA-Z0-9_ ]')
#epatterns.append(r'([A-Za-z. ]+)\s@\s([A-Za-z. ]+)\.edu')
#epatterns.append(r'([A-Za-z. ]+)\sat\s(cs dot stanford)\sedu')
#epatterns.append(r'([A-Za-z.]+)\sAT\s([A-Za-z.;]+)\sDOT\sedu')


# In[436]:


ppatterns = []
ppatterns.append(r'(\d{3})-(\d{3})-(\d{4})')
ppatterns.append(r'.?(\d{3})[^0-9](\d{3})[^0-9](\d{4})')
ppatterns.append(r'.+(\d{3}).[^0-9](\d{3})[^0-9](\d{4})')
#ppatterns.append(r'.+(\d{3}).[^0-9](\d{3})([^0-9a-z&]{8})')


# In[311]:


def process_file(name, f):
    # note that debug info should be printed to stderr
    # sys.stderr.write('[process_file]\tprocessing file: %s\n' % (path))
    res = []
    for line in f:
        # you may modify the line, using something like substitution
        #    before applying the patterns

        # email pattern list
        for epat in epatterns:
            # each epat has 2 sets of parentheses so each match will have 2 items in a list
            matches = re.findall(epat,line)
            for m in matches:
                # string formatting operator % takes elements of list m
                #   and inserts them in place of each %s in the result string
                # email has form  someone@somewhere.edu
                #email = '%s@%s.edu' % m
                email = '{}@{}.edu'.format(m[0],m[1])
                res.append((name,'e',email))

        # phone pattern list
        for ppat in ppatterns:
            # each ppat has 3 sets of parentheses so each match will have 3 items in a list
            matches = re.findall(ppat,line)
            for m in matches:
                # phone number has form  areacode-exchange-number
                #phone = '%s-%s-%s' % m
                phone = '{}-{}-{}'.format(m[0],m[1],m[2])
                res.append((name,'p',phone))
    return res


# In[312]:


def process_dir(data_path):
    # save complete list of candidates
    guess_list = []
    # save list of filenames
    fname_list = []

    for fname in os.listdir(data_path):
        if fname[0] == '.':
            continue
        fname_list.append(fname)
        path = os.path.join(data_path,fname)
        f = open(path,'r', encoding='latin-1')
        # get all the candidates for this file
        f_guesses = process_file(fname, f)
        guess_list.extend(f_guesses)
    return guess_list, fname_list


# In[313]:


def get_gold(gold_path):
    # get gold answers
    gold_list = []
    f_gold = open(gold_path,'r', encoding='latin-1')
    for line in f_gold:
        gold_list.append(tuple(line.strip().split('\t')))
    return gold_list


# In[314]:


def score(guess_list, gold_list, fname_list):
    guess_list = [(fname, _type, value.lower()) for (fname, _type, value) in guess_list]
    gold_list = [(fname, _type, value.lower()) for (fname, _type, value) in gold_list]
    guess_set = set(guess_list)
    gold_set = set(gold_list)

    # for each file name, put the golds from that file in a dict
    gold_dict = {}
    for fname in fname_list:
        gold_dict[fname] = [gold for gold in gold_list if fname == gold[0]]

    tp = guess_set.intersection(gold_set)
    fp = guess_set - gold_set
    fn = gold_set - guess_set

    pp = pprint.PrettyPrinter()
    #print 'Guesses (%d): ' % len(guess_set)
    #pp.pprint(guess_set)
    #print 'Gold (%d): ' % len(gold_set)
    #pp.pprint(gold_set)

    print ('True Positives (%d): ' % len(tp))
    # print all true positives
    pp.pprint(tp)
    print ('False Positives (%d): ' % len(fp))
    # for each false positive, print it and the list of gold for debugging
    for item in fp:
        fp_name = item[0]
        pp.pprint(item)
        fp_list = gold_dict[fp_name]
        for gold in fp_list:
            s = pprint.pformat(gold)
            print('   gold: ', s)
    print ('False Negatives (%d): ' % len(fn))
    # print all false negatives
    pp.pprint(fn)
    print ('Summary: tp=%d, fp=%d, fn=%d' % (len(tp),len(fp),len(fn)))


# In[315]:


def main(data_path, gold_path):
    guess_list, fname_list = process_dir(data_path)
    gold_list =  get_gold(gold_path)
    score(guess_list, gold_list, fname_list)


# In[443]:


if __name__ == '__main__':
    print ('Assuming ContactFinder.py called in directory with data folder')
    main('/Users/zanealderfer/Downloads/ContactFinder/data/dev', '/Users/zanealderfer/Downloads/ContactFinder/data/devGOLD')


# In[ ]:




