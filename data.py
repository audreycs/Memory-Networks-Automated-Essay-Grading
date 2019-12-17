import re
import pandas as pd
import sys


def load_train_data(set_id: int):
    train_data = pd.read_csv("./data/train.tsv", sep='\t', header=0, encoding='utf-8')
    all_data = train_data
    contents = all_data[all_data['essay_set'] == set_id]['essay'].values
    essay_ids = all_data[all_data['essay_set'] == set_id]['essay_id'].values
    essay_scores = all_data[all_data['essay_set'] == set_id]['domain1_score'].values
    essay_contents = []
    for index, content in enumerate(contents):
        content = clean_str(content)
        essay_contents.append(tokenize(content))
    return essay_contents, list(essay_scores), list(essay_ids)


def load_dev_data(set_id: int):
    dev_data = pd.read_csv("./data/dev.tsv", sep='\t', header=0, encoding='utf-8')
    all_data = dev_data
    contents = all_data[all_data['essay_set'] == set_id]['essay'].values
    essay_ids = all_data[all_data['essay_set'] == set_id]['essay_id'].values
    essay_scores = all_data[all_data['essay_set'] == set_id]['domain1_score'].values
    essay_contents = []
    for index, content in enumerate(contents):
        content = clean_str(content)
        essay_contents.append(tokenize(content))
    return essay_contents, list(essay_scores), list(essay_ids)


def load_test_data(set_id: int):
    test_data = pd.read_csv("./data/test.tsv", sep='\t', header=0, encoding='utf-8')
    contents = test_data[test_data['essay_set'] == set_id]['essay'].values
    essay_ids = test_data[test_data['essay_set'] == set_id]['essay_id'].values
    essay_contents = []
    for index, content in enumerate(contents):
        content = clean_str(content)
        essay_contents.append(tokenize(content))
    return essay_contents, list(essay_ids)


def all_vocab(list1, list2, list3):
    w_set = set()
    for i in range(len(list1)):
        for j in list1[i]:
            if j not in w_set:
                w_set.add(j)
    for i in range(len(list2)):
        for j in list2[i]:
            if j not in w_set:
                w_set.add(j)
    for i in range(len(list3)):
        for j in list3[i]:
            if j not in w_set:
                w_set.add(j)
    return w_set


def tokenize(sent):
    """
    Return the tokens of a sentence including punctuation.
    >> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    >> tokenize('I don't know')
        ['I', 'don', '\'', 'know']
    """
    return [x.strip() for x in re.split('(\W+)', sent) if x.strip()]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def load_glove(w_vocab, token_num=6, dim=50):
    word2vec = []
    word_to_index = {}
    # first word is nil
    word2vec.append([0]*dim)
    count = 1
    with open("./glove/glove."+str(token_num)+"B." + str(dim) + "d.txt", encoding='utf-8') as f:
        for line in f:
            l = line.split()
            word = l[0]
            if word in w_vocab:
                vector = list(map(float, l[1:]))
                word_to_index[word] = count
                word2vec.append(vector)
                count += 1
    print("==> glove is loaded")
    print(f"word2vec total size :{sys.getsizeof(word2vec)/1024} KB")
    return word_to_index, word2vec


def vectorize_data(data, word_to_index, sentence_size):
    E = []
    for essay in data:
        ls = max(0, sentence_size - len(essay))
        wl = []
        count = 0
        for w in essay:
            count += 1
            if count > sentence_size:
                break
            if w in word_to_index:
                wl.append(word_to_index[w])
            else:
                wl.append(0)
        wl += [0]*ls
        E.append(wl)
    return E