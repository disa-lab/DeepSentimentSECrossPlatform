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

random.seed()
root_dir = os.getcwd()

# Provide the word embedding model to be used - one value from the following
# Possible values: GoogleNews, Vasiliki, Biswas
WordEmbeddingModel = "GoogleNews"

if WordEmbeddingModel == "GoogleNews":
    model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
    m = "GN"
    vector_dimension = 300
elif WordEmbeddingModel == "Vasiliki":
    model = gensim.models.KeyedVectors.load_word2vec_format("SO_vectors_200.bin", binary=True)
    m = "MSR"
    vector_dimension = 200
elif WordEmbeddingModel == "Biswas":
    model = gensim.models.KeyedVectors.load_word2vec_format("SO_40mill_F300_E10_W10SS5.bin")
    m = "EE"
    vector_dimension = 300


# summarize vocabulary
word_lists = list(model.index2word)

wordVectors = model.syn0
print('Loaded the word vectors!')


maxSeqLength = 50  # Maximum length of sentence
numDimensions = vector_dimension  # Dimensions for each word vector
Vocab_Size = len(word_lists)

# Provide one value from the following, for Sentiment Analysis DataSet
# Possible values: Lin, combined
DataSet_for_SA = "Combined"

if DataSet_for_SA == "Combined":
    # Combine both csv's and provide the path
    file_path = os.path.join(root_dir, 'CombinedData.csv')
    numNegReviews = 1297
    numNeutralReviews = 3496
    numPosReviews = 707
    d = "combined"
    rand_neg = [128, 129, 130, 130, 130, 130, 130, 130, 130, 130]
    rand_pos = [70, 70, 70, 70, 71, 71, 71, 71, 71, 72]


elif DataSet_for_SA == "Lin":
    # Provide the path for Lin Annotations csv
    file_path = os.path.join(root_dir, 'LinData.csv')
    numNegReviews = 178
    numNeutralReviews = 1191
    numPosReviews = 131
    d = "lin"
    rand_neg = [17, 17, 18, 18, 18, 18, 18, 18, 18, 18]
    rand_pos = [13, 13, 13, 13, 13, 13, 13, 13, 13, 14]


# If sampling, provide the sampling rates
# for example NegSampling = 4 means the negative sentences are duplicated 4 times
# If no sampling, provide 1 for each of the three variables, ex: NegSampling = 1
NegSampling = 1
NeutralSampling = 1
PosSampling = 1


###########-----------------------------------------------------------------------------------###########

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


# code for counting the total number of sentences
num_of_sentences = 0  # total number of sentences in the data set
all_sentences = []
sentence_id_from_csv = []

with open(file_path, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        num_of_sentences = num_of_sentences + 1
        sentence_id_from_csv.append(row["id"])
        all_sentences.append(row["text"])


# code for creating the ids array
# ids array specific for Word Embedding model + SA Dataset
ids = np.zeros((num_of_sentences, maxSeqLength), dtype='int32')

sentence_counter = 0

with open(file_path, "r") as csvfile:
    reader=csv.DictReader(csvfile)
    for row in reader:
        indexCounter = 0
        line=row["text"]
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                ids[sentence_counter][indexCounter] = word_lists.index(word)
            except ValueError:
                ids[sentence_counter][indexCounter] = 399999 #Vector for unkown words
            indexCounter = indexCounter + 1
            if indexCounter >= maxSeqLength:
                break
        sentence_counter = sentence_counter + 1

# Save the ids array for reuse
np.save('idsMatrix_%s_%s' % (m, d), ids)

# Load the ids array
ids = np.load('idsMatrix_%s_%s.npy' % (m, d))

# validating Shape of the ids Array
print(ids.shape)

# Building the RNN, Only input RNN hyperparameters here
num_of_folds = 10  # number of validation folds (this is the number of batches)
batchSize = int(num_of_sentences / num_of_folds)  # number of sentences per batch (num_of_sentences/num_of_folds)
lstmUnits = 30
numClasses = 3
numEpochs = 100

tf.reset_default_graph()

batch_dim = 30

labels = tf.placeholder(tf.float32, [batch_dim, numClasses])
input_data = tf.placeholder(tf.int32, [batch_dim, maxSeqLength])

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
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
# writer = tf.summary.FileWriter(logdir, sess.graph)

temp_sentences = []

# Changes related to the Specific SA dataset
def divide_batch():
    # change these numbers according to your dataset
    no_of_neg_reviews = numNegReviews
    no_of_neutral_reviews = numNeutralReviews
    no_of_pos_reviews = numPosReviews


    neg_review_index = 0
    neutral_review_index = no_of_neg_reviews
    pos_review_index = no_of_neg_reviews + no_of_neutral_reviews

    running_neg = 0
    running_neutral = 0
    running_pos = 0

    batch_list = []
    batch_list_test = []

    for i in range(num_of_folds):
        # Calculate the number of neutral sentence in each batch
        rand_neutral = batchSize - (rand_neg[i] + rand_pos[i])

        label = []
        label_test = []

        # Get the sampling rates
        sneu = NeutralSampling
        sneg = NegSampling
        spos = PosSampling
        
        sizeofarr = int((rand_neg[i] * sneg)) + int(rand_neutral * sneu) + int((rand_pos[i] * spos))
        arr = np.zeros([sizeofarr, maxSeqLength])
        arr_test = np.zeros([batchSize, maxSeqLength])
        counter_test = 0
        counter = 0

        for j in range(rand_neg[i]):
            arr_test[counter_test] = ids[neg_review_index]
            label_test.append([1, 0, 0])
            counter_test = counter_test + 1

            # Creating training batch, similar to creating test batch if no sampling
            # If sampling, training batch created by duplicating sentences
            for nw in range(sneg):
                arr[counter] = ids[neg_review_index]
                label.append([1, 0, 0])
                counter = counter + 1

            # Writing test batch
            temp_sentences.append([sentence_id_from_csv[neg_review_index], all_sentences[neg_review_index], 'Negative'])
            neg_review_index = neg_review_index + 1

        for k in range(rand_neutral):
            arr_test[counter_test] = ids[neutral_review_index]
            label_test.append([0, 1, 0])
            counter_test = counter_test + 1

            if k <= (rand_neutral * sneu)-1:
                arr[counter] = ids[neutral_review_index]
                label.append([0, 1, 0])
                counter = counter + 1

            temp_sentences.append([sentence_id_from_csv[neutral_review_index], all_sentences[neutral_review_index], 'Neutral'])
            neutral_review_index = neutral_review_index + 1


        for l in range(rand_pos[i]):
            arr_test[counter_test] = ids[pos_review_index]
            label_test.append([0, 0, 1])
            counter_test = counter_test + 1

            for pw in range(spos):
                arr[counter] = ids[pos_review_index]
                label.append([0, 0, 1])
                counter = counter + 1

            temp_sentences.append([sentence_id_from_csv[pos_review_index], all_sentences[pos_review_index], 'Positive'])
            pos_review_index = pos_review_index + 1

        running_neg = running_neg + rand_neg[i]
        running_neutral = running_neutral + rand_neutral
        running_pos = running_pos + rand_pos[i]

        print(rand_neg[i], rand_neutral, rand_pos[i])
        batch_list.append([arr, label])
        batch_list_test.append([arr_test, label_test])

        print("Training Array Shape, Label Length")
        print(arr.shape, len(label))
        print("Test Array Shape, Label Length")
        print(arr_test.shape, len(label_test))

    # Write the batches created in the file mentioned below
    with open("output_%s_%s.csv" % (m, d), 'a') as f:
        for list_items in temp_sentences:
            f.write("%s\t%s\t%s\n" % (list_items[0], list_items[1], list_items[2]))

    return batch_list, batch_list_test


def divide_sub_batches(temp_batch, index_start, index_end):
    return temp_batch[0][index_start:index_end], temp_batch[1][index_start:index_end]


def write_log(s):
    temp_data = []
    for p in s:
        for items in p:
            a = list(items)
            position = a.index(max(a))
            temp_data.append(position)

    # Write the predictions in the file mentioned below
    with open("actualpredictions_%s_%s.csv" % (m, d), 'a') as f:
        for list_items in temp_data:
            f.write("%s\n" % ([list_items]))
    return


train_batches, test_batches = divide_batch()

for fold in range(num_of_folds):
    test_batch_index = 0
    res = 0
    acc_stack = []
    pred_stack = []
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # Run according to the number of epochs:
    for epochx in range(50):
        for x in range(len(train_batches)):
            if x != fold:
                nooftimes_alltrainsam = int(len(train_batches[x][0]) / batch_dim) + 1
                remain_smp = int(len(train_batches[x][0]) % batch_dim)
                addreqd = batch_dim - remain_smp

                # print(nooftimes_alltrainsam, remain_smp)

                for w in range(nooftimes_alltrainsam):
                    if w == nooftimes_alltrainsam - 1:
                        if remain_smp == 0:
                            print("last loop not required")
                        else:
                            next_batch, next_label = divide_sub_batches(train_batches[x], 30 * w - addreqd, 30 * w + remain_smp)
                            sess.run(optimizer, {input_data: next_batch, labels: next_label, Wembed: wordVectors})
                    else:
                        next_batch, next_label = divide_sub_batches(train_batches[x], 30 * w, 30 * w + 30)
                        sess.run(optimizer, {input_data: next_batch, labels: next_label, Wembed: wordVectors})
            else:
                test_batch_index = x
    try:
        nooftimes_alltestsam = int(len(test_batches[test_batch_index][0]) / batch_dim) + 1
        remain_smp_test = int(len(test_batches[test_batch_index][0]) % batch_dim)
        addreqd_test = batch_dim - remain_smp_test
        for y in range(nooftimes_alltestsam):
            if y == nooftimes_alltestsam - 1:
                if remain_smp_test == 0:
                    print("last loop test not required")
                else:
                    next_test_batch, next_test_label = divide_sub_batches(test_batches[test_batch_index], 30 * y - addreqd_test,
                                                                          30 * y + remain_smp_test)
                    res, predictx_test = sess.run([accuracy, prediction],
                                             {input_data: next_test_batch, labels: next_test_label,
                                              Wembed: wordVectors})
                    predictx =[]
                    for z in range(30):
                        if z < addreqd_test:
                            continue
                        else:
                            predictx.append(predictx_test[z])
                    pred_stack.append(predictx)
            else:
                next_test_batch, next_test_label = divide_sub_batches(test_batches[test_batch_index], 30 * y, 30 * y + 30)
                res, predictx = sess.run([accuracy, prediction], {input_data: next_test_batch, labels: next_test_label, Wembed: wordVectors})
                acc_stack.append(res)
                pred_stack.append(predictx)
    except:
        print(test_batches[test_batch_index][0], test_batches[test_batch_index][1])

    # print(acc_stack)
    write_log(pred_stack)
    sess.close()

