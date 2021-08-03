import numpy as np
import os
import gensim
import logging
import tensorflow as tf
import re
import csv
import random
from random import randint
import datetime
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold



random.seed(10)
root_dir = "/home/mdabdullahal.alamin/alamin/sentiment/cross_platform/rnn/"
os.chdir(root_dir)


WordEmbeddingModel = "GoogleNews"
print("Going to load Gensim Pakage")
if WordEmbeddingModel == "GoogleNews":
    model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
    m = "GN"
    vector_dimension = 300

print("Going to word_lists")
# summarize vocabulary
word_lists = list(model.index2word)
wordVectors = model.syn0
print('Loaded the word vectors!')












maxSeqLength = 50  # Maximum length of sentence
numDimensions = vector_dimension  # Dimensions for each word vector
Vocab_Size = len(word_lists)


# Building the RNN, Only input RNN hyperparameters here
# num_of_folds = 10  # number of validation folds (this is the number of batches)
# batchSize = int(num_of_sentences / num_of_folds)  # number of sentences per batch (num_of_sentences/num_of_folds)
# batchSize = 6000
lstmUnits = 30
numClasses = 3
numEpochs = 100

tf.reset_default_graph()
batch_dim = 30

labels = tf.placeholder(tf.float32, [batch_dim, numClasses])
input_data = tf.placeholder(tf.int32, [batch_dim, maxSeqLength])

# labels = tf.placeholder(tf.float32, [None, numClasses])
# input_data = tf.placeholder(tf.int32, [None, maxSeqLength])


# Put the vocab size and the vector dimension in here
Wembed = tf.placeholder(tf.float32, [Vocab_Size, numDimensions], name='Wembed')

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={Wembed: wordVectors})

data = tf.Variable(tf.zeros([batch_dim, maxSeqLength, numDimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(Wembed, input_data)
print(data.shape)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
# Input Desired dropout rate
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.8)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()




def disa_create_train_test_set(input_file, dataset, fold):
    dataset = dataset.lower()
    df_all = pd.read_csv(input_file, dtype={'oracle': str})
    df_all = df_all[['id', 'text', 'oracle', 'dataset']]

    if(dataset=='all'):
        dataset_df = df_all
        dataset_test = "_" + str(fold)
        test_df = dataset_df[dataset_df['dataset'].str.lower().str.contains(dataset_test)]
    elif(dataset=='lin' or dataset=='new'):
        dataset_df = df_all[df_all['dataset'].astype(str).str.lower().str.contains(dataset)]
        dataset_test = dataset + "_" + str(fold)
        test_df = dataset_df[dataset_df['dataset'].str.lower() == dataset_test]
    else:
        dataset_df = df_all[df_all['dataset'].astype(str).str.lower().str.contains(dataset)]
        # print("lenght of the dataset %s is : %d"% (dataset, len(dataset_df)))
        dataset_test = dataset + "_test_" + str(fold)
        if(dataset == "datasetlinjira"):
            dataset_test = dataset + "_cleaned_test_" + str(fold)
        test_df = dataset_df[dataset_df['dataset'].str.lower() == dataset_test]
    
    
    
    test_ids = test_df['id'].tolist()
    train_df = dataset_df[~dataset_df['id'].isin(test_ids)]

    train_df = train_df.drop('dataset', axis = 1) # 0 means rows 1 means column
    test_df = test_df.drop('dataset', axis = 1) # 0 means rows 1 means column

    print("len of test_df %d and len of train_df %d"%(len(test_df), len(train_df)))
    assert len(train_df) + len(test_df) == len(dataset_df)

    train_ids = train_df['id'].tolist()
    train_texts = train_df['text'].tolist()
    train_label = train_df['oracle'].tolist()

    test_ids = test_df['id'].tolist()
    test_texts = test_df['text'].tolist()
    test_label = test_df['oracle'].tolist()

    return train_ids, train_texts, train_label, test_ids, test_texts, test_label


def disa_create_cross_platform_train_test_set(input_file, train_dataset):
    train_dataset = train_dataset.lower()
    # test_dataset = test_dataset.lower()
    df_all = pd.read_csv(input_file, dtype={'oracle': str})
    df_all = df_all[['id', 'text', 'oracle', 'dataset']]


    train_df = df_all[df_all['dataset'].astype(str).str.lower().str.contains(train_dataset)]
    train_ids = train_df['id'].tolist()
    test_df = df_all[~df_all['id'].isin(train_ids)]

    train_df = train_df.drop('dataset', axis = 1) # 0 means rows 1 means column
    test_df = test_df.drop('dataset', axis = 1) # 0 means rows 1 means column


    print("len of test_df %d and len of train_df %d"%(len(test_df), len(train_df)))
    assert len(train_df) + len(test_df) == len(df_all)

    train_ids = train_df['id'].tolist()
    train_texts = train_df['text'].tolist()
    train_label = train_df['oracle'].tolist()

    test_ids = test_df['id'].tolist()
    test_texts = test_df['text'].tolist()
    test_label = test_df['oracle'].tolist()

    return train_ids, train_texts, train_label, test_ids, test_texts, test_label



strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def text_embedding_from_ids(id_list, map_embedding, maxSeqLength=50):
    # result = np.zeros((len(texts), maxSeqLength), dtype='int32')
    result = np.zeros((len(id_list), maxSeqLength), dtype='int32')
    for ind, id in enumerate(id_list):
        # print(ind, id)
        result[ind] = map_embedding[id]
    return result

def text_embedded_single(text, maxSeqLength = 50):
    result = np.zeros(maxSeqLength, dtype='int32')
    cleaned_text = cleanSentences(text)
    split = cleaned_text.split()
    for ind, word in enumerate(split):
        try:
            result[ind] = word_lists.index(word)
        except ValueError:
            result[ind] = 399999 #Vector for unkown words
        if ind >= maxSeqLength-1:
            break
    return result


def label_embedded(labels):
    result = []
    for label in labels:
        if(str(label) == '-1'):
            result.append([1, 0, 0])
        elif(str(label) == '0'):
            result.append([0, 1, 0])
        elif(str(label) == '1'):
            result.append([0, 0, 1])
    return result

def get_predictions(preds):
    result = []
    pred_to_oracle = [-1, 0, 1]
    for pred in preds:
        pred_list = list(pred)
        ind = pred_list.index(max(pred_list))
        result.append(pred_to_oracle[ind])
    return result

def get_batches(data_list, batch_size=30):
    result = [data_list[x:x+batch_size] for x in range(0, len(data_list), batch_size)]
    last_batch = result[-1]
    last_batch_size = len(last_batch)
    # print("org last batch %d %s " % (len(result[-1]), result[-1][-1]))
    if(len(last_batch) != batch_size):
        # print("Batch Size is not equal so copying some samples from first batch")
        temp_batch = result[0].copy()
        # print(temp_batch)
        for ind in range(last_batch_size):
            temp_batch[ind] = last_batch[ind]
        result[-1] = temp_batch
    return result




# MAP_EMBEDDING = np.load("processed_word_embedding.npy", allow_pickle='TRUE').item()
MAP_EMBEDDING = np.load("processed_word_embedding_cross_platform.npy", allow_pickle='TRUE').item()
print("processed map is loaded from the file system. Len of map %d " % (len(MAP_EMBEDDING)))
# print(MAP_EMBEDDING[238])
# print(map_embedding[238])



dataset_file = "/home/mdabdullahal.alamin/alamin/sentiment/cross_platform/dataset/processed/combined.csv"
datasets = ["DatasetSenti4SDSO", "OrtuJIRA", "GitHub"]
print(datetime.datetime.now())
num_of_epochs = 160
num_of_folds = 10
# all_preds = []

for dataset in datasets:
    DataSet_for_SA = dataset.lower()
    for fold in range(num_of_folds):
        print ("############## Dataset %s Fold %d #################" % (DataSet_for_SA, fold))
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())


        train_ids, train_texts, train_label, test_ids, test_texts, test_label = \
        disa_create_train_test_set(input_file=dataset_file, dataset=DataSet_for_SA, fold=fold)

        print("Stat for training")
        print(Counter(train_label).keys())
        print(Counter(train_label).values())

        print("Stat for test")
        print(Counter(test_label).keys())
        print(Counter(test_label).values())

        train_texts_embed = text_embedding_from_ids(train_ids, map_embedding=MAP_EMBEDDING)
        test_texts_embed = text_embedding_from_ids(test_ids, map_embedding=MAP_EMBEDDING)
        train_label_embed = label_embedded(train_label)
        test_label_embed = label_embedded(test_label)

        # # Training
        # for epochx in range(num_of_epochs):
        #     sess.run(optimizer, {input_data: train_texts_embed, labels: train_label_embed, Wembed: wordVectors})

        print("# Using batch embedding code to train now")
        # This code will not work unless the training sample size is divisible by 30. Need to write better code
        # the one written in the testing part
        
        train_texts_embed_batches =  get_batches(train_texts_embed)
        train_label_embed_batches =  get_batches(train_label_embed)
        
        for epochx in range(num_of_epochs):
            for batch in range(len(train_texts_embed_batches)):
                next_batch = train_texts_embed_batches[batch]
                next_label = train_label_embed_batches[batch]
                sess.run(optimizer, {input_data: next_batch, labels: next_label, Wembed: wordVectors})

        
        # # Testing
        # res, predictx = sess.run([accuracy, prediction], {input_data: test_texts_embed, labels: test_label_embed, Wembed: wordVectors})
        # print("Accurary is: %s " % (res))

        print("# Using batch embedding code to test now with unreadable code")
        # Wrting this shitty code in order to ensure batch size in the test cases. May be fix this code later
        

        test_texts_embed_batches =  get_batches(test_texts_embed)
        test_label_embed_batches =  get_batches(test_label_embed)
        total_tests = len(test_ids)
        # predictx = []
        preds = []
        for i in range (len(test_texts_embed_batches)):
            next_test_text = test_texts_embed_batches[i]
            next_test_label = test_label_embed_batches[i]
            res, pred = sess.run([accuracy, prediction], {input_data: next_test_text, labels: next_test_label, Wembed: wordVectors})
            # predictx.append(pred)
            preds += get_predictions(pred)
        preds = preds[:total_tests]
        print("# of tests %d and #preds %d" % (len(test_label_embed), len(preds)))
        assert len(test_label_embed) == len(preds)





        ## writing result to the output/out directory. make sure this directory exists
        name =  DataSet_for_SA + "_" + str(fold) + ".csv"
        out_dir = "output/inner/"
        if(not os.path.exists(out_dir)):
            os.makedirs(out_dir)
        output_file = os.path.join(out_dir, name)
        # preds = get_predictions(predictx)
        
        df = pd.DataFrame()
        df['id'] = test_ids
        df['pred_label'] = preds
        df['oracle'] = test_label
        df.to_csv(output_file, index=False)
        print("output has been saved to %s " % (output_file))
        print(datetime.datetime.now())

    sess.close()
