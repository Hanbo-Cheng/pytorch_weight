import pickle as pkl
import sys
import numpy as np


def tap_dataIterator(feature_file, label_file, align_file, dictionary, batch_size, maxlen):
    fp = open(feature_file, 'rb')  # read kaldi scp file
    features = pkl.load(fp)  # load features in dict
    fp.close()

    fp2 = open(label_file, 'r')
    labels = fp2.readlines()
    fp2.close()

    fp3 = open(align_file, 'rb')
    aligns = pkl.load(fp3)
    fp3.close()

    # **********************************Symbol classify 's label**********************************

    targets = {}
    # map word to int with dictionary
    for l in labels:
        tmp = l.strip().split()
        uid = tmp[0]
        w_list = []
        for w in tmp[1:]:
            if w in dictionary:
                w_list.append(dictionary[w])
            else:
                print('a word not in the dictionary !! sentence ', uid, 'word ', w)

                sys.exit()
        targets[uid] = w_list
    # **********************************************************************************************
    # ××××××××××××××××××××××××××××××××××××××××收集所有样例拥有坐标点数并排序××××××××××××××××××××××××××××××
    sentLen = {}
    for uid, fea in features.items():
        sentLen[uid] = len(fea)

    sentLen = sorted(sentLen.items(),
                     key=lambda d: d[1])  # sorted by sentence length,  return a list with each triple element
    # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    # ××××××××××××××××××××××××××××××××××××按照坐标点数排序生成batch××××××××××××××××××××××××××××××××××××××
    feature_batch = []
    label_batch = []
    alignment_batch = []
    feature_total = []
    label_total = []
    alignment_total = []

    i = 0
    max_length_fea = -1
    min_length_fea = 9999999999
    for uid, length in sentLen:
        fea = features[uid]

        if len(fea) > max_length_fea:
            max_length_fea = len(fea)
        if len(fea) < min_length_fea:
            min_length_fea = len(fea)
        ali = aligns[uid]
        lab = targets[uid]

        if len(lab) > maxlen:
            print('sentence', uid, 'y length bigger than', maxlen, 'ignore')
        else:
            if i == batch_size:  # a batch is full
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                alignment_total.append(alignment_batch)

                i = 0
                feature_batch = []
                label_batch = []
                alignment_batch = []
                feature_batch.append(fea)
                label_batch.append(lab)
                alignment_batch.append(ali)
                i = i + 1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                alignment_batch.append(ali)
                i = i + 1

    # last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    alignment_total.append(alignment_batch)
    # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    print('total ', len(feature_total), 'batch data loaded')
    print('最长特征点 ', max_length_fea)
    print('最短特征点 ', min_length_fea)
    new_total = []
    for i, x in enumerate(feature_total):
        new_total.append((x, label_total[i], alignment_total[i]))

    return new_total, len(feature_total)


def tap_dataIterator_valid(feature_file, label_file, dictionary, batch_size, maxlen):
    fp = open(feature_file, 'rb')  # read kaldi scp file
    features = pkl.load(fp)  # loaatures in dict
    fp.close()

    fp2 = open(label_file, 'r')
    labels = fp2.readlines()
    fp2.close()

    targets = {}
    # map word to int with dictionary
    for l in labels:
        tmp = l.strip().split()
        uid = tmp[0]
        w_list = []
        for w in tmp[1:]:
            if w in dictionary:
                w_list.append(dictionary[w])
            else:
                print('a word not in the dictionary !! sentence ', uid, 'word ', w)
                sys.exit()
        targets[uid] = w_list

    # targets['23_em_01'] = \frac { 8 } { 7 }
    sentLen = {}
    for uid, fea in features.items():
        sentLen[uid] = len(fea)

    sentLen = sorted(sentLen.items(),
                     key=lambda d: d[1])  # sorted by sentence length,  return a list with each triple element

    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    uidList = []

    i = 0
    for uid, length in sentLen:
        fea = features[uid]
        lab = targets[uid]

        if len(lab) > maxlen:
            print('sentence', uid, 'length bigger than', maxlen, 'ignore')
        else:
            uidList.append(uid)
            if i == batch_size:  # a batch is full
                feature_total.append(feature_batch)
                label_total.append(label_batch)

                i = 0
                feature_batch = []
                label_batch = []
                feature_batch.append(fea)
                label_batch.append(lab)
                i = i + 1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i = i + 1

    # last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)

    print('total ', len(feature_total), 'batch data loaded')

    new_total = []
    for i, x in enumerate(feature_total):
        new_total.append((x, label_total[i]))

    return new_total, uidList


# batch preparation
def tap_prepare_data(params, seqs_x, seqs_y, seqs_a, maxlen=None, n_words_src=30000,
                     n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_seqs_a = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y, s_a in zip(lengths_x, seqs_x, lengths_y, seqs_y, seqs_a):
            if l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
                new_seqs_a.append(s_a)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y
        seqs_a = new_seqs_a

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1
    maxlen_y = np.max(lengths_y) + 1

    x = np.zeros((maxlen_x, n_samples, params['dim_feature'])).astype('float32')  # SeqX * batch * dim
    y = np.zeros((maxlen_y, n_samples)).astype('int64')  # the <eol> must be 0 in the dict !!!
    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')
    a = np.zeros((maxlen_x, n_samples, maxlen_y)).astype('float32')  # SeqX * batch * SeqY
    a_mask = np.zeros((maxlen_x, n_samples, maxlen_y)).astype('float32')  # SeqX * batch * SeqY
    for idx, [s_x, s_y, s_a] in enumerate(zip(seqs_x, seqs_y, seqs_a)):
        x[:lengths_x[idx], idx, :] = s_x  # the zeros frame is a padding frame to align <eol>
        x_mask[:lengths_x[idx] + 1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.
        a[:lengths_x[idx], idx, :lengths_y[idx]] = s_a * 1.
        a_mask[:lengths_x[idx] + 1, idx, :lengths_y[idx] + 1] = 1.

    return x, x_mask, y, y_mask, a, a_mask
