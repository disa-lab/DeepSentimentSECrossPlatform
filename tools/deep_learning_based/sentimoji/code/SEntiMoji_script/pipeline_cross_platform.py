"""
Pipeline code for training and evaluating the sentiment classifier.
We use the Deepmoji architecture here, see https://github.com/bfelbo/DeepMoji for detail.
"""
import re
import codecs
import random
import numpy as np
import sys
import json
import argparse
import pandas as pd
import glob, os
import matplotlib.pylab as plt

sys.path.append("DeepMoji/deepmoji/")

from sentence_tokenizer import SentenceTokenizer
from model_def import deepmoji_architecture, load_specific_weights
from finetuning import load_benchmark, finetune

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

MAX_LEN = 150


# def load_data(filename):
#     f = codecs.open(filename, "r", "utf-8")
#     data_pair = []
#     for line in f:
#         line = line.strip().split("\t")
#         line = line.strip().split(",")
#         data_pair.append((line[0], line[1]))
#     return data_pair

# def load_data(filename):
#     df = pd.read_csv(filename, sep="\t")
#     data_pair = []
#     for index, row in df.iterrows():
#         data_pair.append((row[0], row[1], row[2]))
#     return data_pair


# def prepare_5fold(data_pair):
#     sind = 0
#     eind = 0
#     random.shuffle(data_pair)
#     fold_size = int(len(data_pair) / 5)
#     for fold in range(0, 5):
#         sind = eind
#         eind = sind + fold_size
#         train_pair = data_pair[0:sind] + data_pair[eind:len(data_pair)]
#         test_pair = data_pair[sind:eind]
#         yield (train_pair, test_pair)


def get_train_test_data(infile, dataset, fold):


    dataset = dataset.lower()
    df_all = pd.read_csv(input_file, usecols=['id', 'dataset', 'text', 'oracle'], dtype={'oracle': str})
    df_all = df_all[['id', 'text', 'oracle', 'dataset']]
    print("len df_all %d" % len(df_all))

    dataset_df = df_all[df_all['dataset'].astype(str).str.lower().str.contains(dataset)]
    print("lenght of the dataset %s is : %d"% (dataset, len(dataset_df)))
    dataset_test = dataset + "_test_" + str(fold)
    if(dataset == "datasetlinjira"):
        dataset_test = dataset + "_cleaned_test_" + str(fold)
    test_df = dataset_df[dataset_df['dataset'].str.lower() == dataset_test]
    test_ids = test_df['id'].tolist()
    train_df = dataset_df[~dataset_df['id'].isin(test_ids)]
    print("len test_df %d " % (len(test_df)))

    train_df = train_df.drop('dataset', axis = 1) # 0 means rows 1 means column
    test_df = test_df.drop('dataset', axis = 1) # 0 means rows 1 means column

    print("len of test_df %d and len of train_df %d"%(len(test_df), len(train_df)))
    assert len(train_df) + len(test_df) == len(dataset_df)

    train_pair = []
    test_pair = []
    for index, row in train_df.iterrows():
            train_pair.append((row['id'], row['text'], row['oracle']))
    for index, row in test_df.iterrows():
            test_pair.append((row['id'], row['text'], row['oracle']))

    return train_pair, test_pair


def get_train_test_cross_platform_data(input_file, train_dataset):
    train_dataset = train_dataset.lower()
    # test_dataset = test_dataset.lower()
    df_all = pd.read_csv(input_file, usecols=['id', 'dataset', 'text', 'oracle'], dtype={'oracle': str})
    df_all = df_all[['id', 'text', 'oracle', 'dataset']]


    train_df = df_all[df_all['dataset'].astype(str).str.lower().str.contains(train_dataset)]
    train_ids = train_df['id'].tolist()
    test_df = df_all[~df_all['id'].isin(train_ids)]

    train_df = train_df.drop('dataset', axis = 1) # 0 means rows 1 means column
    test_df = test_df.drop('dataset', axis = 1) # 0 means rows 1 means column

    print("Train dataset: %s and len %d " % ( train_dataset, len(train_df)))
    print("Test dataset: len %d " % ( len(test_df)))
    assert len(train_df) + len(test_df) == len(df_all)

    train_pair = []
    test_pair = []
    for index, row in train_df.iterrows():
            train_pair.append((row['id'], row['text'], row['oracle']))
    for index, row in test_df.iterrows():
            test_pair.append((row['id'], row['text'], row['oracle']))

    return train_pair, test_pair




def get_train_test(infile, dataset, fold):
    if(fold != None):
        train_pair, test_pair = get_train_test_data(infile=input_file, dataset = dataset, fold=fold)
    else:
        train_pair, test_pair = get_train_test_cross_platform_data(input_file=infile, train_dataset = dataset)
    train_id = [p[0] for p in train_pair]
    train_text = [str(p[1]) for p in train_pair]
    train_label = [str(p[2]) for p in train_pair]

    test_id = [p[0] for p in test_pair]
    test_text = [str(p[1]) for p in test_pair]
    test_label = [str(p[2]) for p in test_pair]

    return train_id, train_text, train_label, test_id, test_text, test_label

if __name__ == "__main__":
    print("Alamin")
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=["SEntiMoji", "SEntiMoji-T", "SEntiMoji-G"], help="name of pretrained representation model")
    parser.add_argument("--task", type=str.lower, required=True, choices=["sentiment", "emotion"], help="specify task (sentiment or emotion)")
    parser.add_argument("--benchmark_dataset_name", type=str, required=False, choices=["Jira", "StackOverflow", "CodeReview", "JavaLib"], help="name of benchmark dataset")
    parser.add_argument("--emotion_type", type=str.lower, required=False, default=None, choices=["anger", "love", "deva", "joy", "sad"], help="specify emotion dataset")
    
    parser.add_argument("--use_own_dataset", action='store_true', help="whether use your own dataset or not")
    parser.add_argument("--own_dataset_dir", type=str, required=False, default=None, help="directory of your train data file")
    parser.add_argument("--own_dataset_file", type=str, required=False, default=None, help="file name of your train data file")
    parser.add_argument("--cross_platform", action='store_true', help="This will load code to run against cross platform dataset")

    args = parser.parse_args()

    print("args:")
    d = args.__dict__
    for key,value in d.items():
        print("%s = %s"%(key,value))

    # parse arguments
    model_path = "../../model/representation_model/model_%s.hdf5" % args.model
    vocab_path = "vocabulary/vocabulary_%s.json" % args.model
    out_dir = "../../output/out/"
    base_dir = "/home/mdabdullahal.alamin/alamin/sentiment/sentimoji"

    # load vocabulary 
    with open(vocab_path, "r") as f_vocab:
        vocabulary = json.load(f_vocab)


    label2index = {"0": 0, "1": 1, "-1": 2}
    index2label = {i: l for l, i in label2index.items()}



    # sentence tokenizer (MAXLEN means the max length of input text)
    st = SentenceTokenizer(vocabulary, MAX_LEN)
    fold = 0

    datasets = ["DatasetSenti4SDSO", "OrtuJIRA", "GitHub"]
    # datasets = ["DatasetSenti4SDSO"]
    input_file ="/home/mdabdullahal.alamin/alamin/sentiment/cross_platform/dataset/processed/combined.csv"

    
    test_type = "inner_platform" # test_type can be inner_platform or cross platform
    # test_type = "cross_platform"
    test_type = "both"

    if(test_type == "cross_platform" or test_type == "both"):
        print("============> Cross platform train test")
        for dataset in datasets:
            dataset = dataset.lower()
            train_id, train_text, train_label, test_id, test_text, test_label = get_train_test(infile=input_file, dataset = dataset, fold=None)
            # print(type(train_text[0]))
            print("len train: %d and len test %d"%(len(train_id), len(test_id)))
            train_X, _, _ = st.tokenize_sentences(train_text)
            test_X, _, _ = st.tokenize_sentences(test_text)
            train_y = np.array([label2index[l] for l in train_label])
            test_y = np.array([label2index[l] for l in test_label])

            nb_classes = len(label2index)
            nb_tokens = len(vocabulary)

            # use 20% of the training set for validation
            train_X, val_X, train_y, val_y = train_test_split(train_X, train_y,
                                                            test_size=0.2, random_state=0)
            # # model 
            model = deepmoji_architecture(nb_classes=nb_classes,
                                        nb_tokens=nb_tokens,
                                        maxlen=MAX_LEN, embed_dropout_rate=0.25, final_dropout_rate=0.5, embed_l2=1E-6)
            model.summary()

            # # load pretrained representation model
            load_specific_weights(model, model_path, nb_tokens, MAX_LEN,
                                exclude_names=["softmax"])
            # 
            # # train model
            model, acc = finetune(model, [train_X, val_X, test_X], [train_y, val_y, test_y], nb_classes, 100,
                                method="chain-thaw", verbose=0)
            
            pred_y_prob = model.predict(test_X)

            if nb_classes == 2:
                pred_y = [0 if p < 0.5 else 1 for p in pred_y_prob]
            else:
                pred_y = np.argmax(pred_y_prob, axis=1)

            # evaluation
            print("*****************************************")
            # print("Fold %d" % fold)
            accuracy = accuracy_score(test_y, pred_y)
            print("Accuracy: %.3f" % accuracy)

            precision = precision_score(test_y, pred_y, average=None)
            recall = recall_score(test_y, pred_y, average=None)
            f1score = f1_score(test_y, pred_y, average=None)
            labels = list(set(test_y))
            precision = precision_score(test_y, pred_y, average=None, labels = labels)
            recall = recall_score(test_y, pred_y, average=None, labels = labels)
            f1score = f1_score(test_y, pred_y, average=None, labels = labels)

            for index in range(0, len(labels)):
                print("label: %s" % index2label[index])
                print("Precision: %.3f, Recall: %.3f, F1 score: %.3f" % (precision[index], recall[index], f1score[index]))
            print("*****************************************")

            save_name = "sentimoji_train_%s.csv" % (dataset)       
            save_name = os.path.join(out_dir, save_name)
            # if(not os.path.exists(save_name)):
            #     os.makedirs(save_name)
            with open(save_name, "w", encoding="utf-8") as f:
                for i in range(0, len(test_text)):
                    f.write("%s,%s\r\n" % (test_id[i], index2label[pred_y[i]]))
            print("#%d test results has been saved to: %s" % (len(test_text), save_name))

            output_dir = "../../model/trained_model" + str(fold) + ".h5"


    if(test_type == "inner_platform" or test_type == "both"):
        print("============> Inner platform train test")
        for dataset in datasets:
            dataset = dataset.lower()
            for fold in range(10):
                train_id, train_text, train_label, test_id, test_text, test_label = get_train_test(infile=input_file, dataset = dataset, fold=fold)
                # print(type(train_text[0]))
                print("len train: %d and len test %d"%(len(train_id), len(test_id)))
                train_X, _, _ = st.tokenize_sentences(train_text)
                test_X, _, _ = st.tokenize_sentences(test_text)
                train_y = np.array([label2index[l] for l in train_label])
                test_y = np.array([label2index[l] for l in test_label])

                nb_classes = len(label2index)
                nb_tokens = len(vocabulary)

                # use 20% of the training set for validation
                train_X, val_X, train_y, val_y = train_test_split(train_X, train_y,
                                                                test_size=0.2, random_state=0)
                # # model 
                model = deepmoji_architecture(nb_classes=nb_classes,
                                            nb_tokens=nb_tokens,
                                            maxlen=MAX_LEN, embed_dropout_rate=0.25, final_dropout_rate=0.5, embed_l2=1E-6)
                model.summary()

                # # load pretrained representation model
                load_specific_weights(model, model_path, nb_tokens, MAX_LEN,
                                    exclude_names=["softmax"])
                # 
                # # train model
                model, acc = finetune(model, [train_X, val_X, test_X], [train_y, val_y, test_y], nb_classes, 100,
                                    method="chain-thaw", verbose=0)

                
                pred_y_prob = model.predict(test_X)

                if nb_classes == 2:
                    pred_y = [0 if p < 0.5 else 1 for p in pred_y_prob]
                else:
                    pred_y = np.argmax(pred_y_prob, axis=1)

                # evaluation
                print("*****************************************")
                print("Fold %d" % fold)
                accuracy = accuracy_score(test_y, pred_y)
                print("Accuracy: %.3f" % accuracy)

                precision = precision_score(test_y, pred_y, average=None)
                recall = recall_score(test_y, pred_y, average=None)
                f1score = f1_score(test_y, pred_y, average=None)
                labels = list(set(test_y))
                precision = precision_score(test_y, pred_y, average=None, labels = labels)
                recall = recall_score(test_y, pred_y, average=None, labels = labels)
                f1score = f1_score(test_y, pred_y, average=None, labels = labels)

                for index in range(0, len(labels)):
                    print("label: %s" % index2label[index])
                    print("Precision: %.3f, Recall: %.3f, F1 score: %.3f" % (precision[index], recall[index], f1score[index]))
                print("*****************************************")

                # save predict result

                save_name = "sentimoji_%s_%d.csv" % ( dataset, fold)        
                save_name = os.path.join(out_dir, save_name)
                # if(not os.path.exists(save_name)):
                #     os.makedirs(save_name)
                with open(save_name, "w", encoding="utf-8") as f:
                    for i in range(0, len(test_text)):
                        f.write("%s,%s\r\n" % (test_id[i], index2label[pred_y[i]]))
                print("#%d test results has been saved to: %s" % (len(test_text), save_name))
                fold += 1
                # output_dir = "../../model/trained_model" + str(fold) + ".h5"
                # break
            # break
print("Alamin Over")