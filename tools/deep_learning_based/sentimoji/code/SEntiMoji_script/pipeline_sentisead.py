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

def load_data(filename):
    df = pd.read_csv(filename, sep="\t")
    data_pair = []
    for index, row in df.iterrows():
        data_pair.append((row[0], row[1], row[2]))
    return data_pair


def prepare_5fold(data_pair):
    sind = 0
    eind = 0
    random.shuffle(data_pair)
    fold_size = int(len(data_pair) / 5)
    for fold in range(0, 5):
        sind = eind
        eind = sind + fold_size
        train_pair = data_pair[0:sind] + data_pair[eind:len(data_pair)]
        test_pair = data_pair[sind:eind]
        yield (train_pair, test_pair)


def get_train_test_data(infile, dataset, fold):

    df_all = pd.read_excel(input_file, sheet_name="Sheet1", usecols="S, AF, T, AX", 
     names=['dataset', 'oracle', 'text', 'id'])
    # df_all.insert(loc=0, column="id", value=df_all.index + 1)
    # df_all['id'] = df_all.index
    df_all = df_all[['id', 'text', 'oracle', 'dataset']]
    # print("length of all datasets %d" % len(df_all))

    df_all.loc[df_all.oracle == 'o', 'oracle'] = '0'
    df_all.loc[df_all.oracle == 'n', 'oracle'] = '-1'
    df_all.loc[df_all.oracle == 'p', 'oracle'] = '1'
    # print(df_all.columns)

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

    train_pair = []
    test_pair = []
    for index, row in train_df.iterrows():
            train_pair.append((row['id'], row['text'], row['oracle']))
    for index, row in test_df.iterrows():
            test_pair.append((row['id'], row['text'], row['oracle']))

    # dataset_dir =  "/home/mdabdullahal.alamin/alamin/sentiment/bert/dataset/"
    # train_df.to_csv( dataset_dir + "train.tsv", sep='\t', index=False, header = None)
    # test_df.to_csv( dataset_dir + "test.tsv", sep='\t', index=False, header = None)

    return train_pair, test_pair


def get_train_test(infile, dataset, fold):
    train_pair, test_pair = get_train_test_data(infile=input_file, dataset = dataset, fold=fold)
    train_id = [p[0] for p in train_pair]
    train_text = [str(p[1]) for p in train_pair]
    train_label = [str(p[2]) for p in train_pair]

    test_id = [p[0] for p in test_pair]
    test_text = [str(p[1]) for p in test_pair]
    test_label = [str(p[2]) for p in test_pair]

    return train_id, train_text, train_label, test_id, test_text, test_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=["SEntiMoji", "SEntiMoji-T", "SEntiMoji-G"], help="name of pretrained representation model")
    parser.add_argument("--task", type=str.lower, required=True, choices=["sentiment", "emotion"], help="specify task (sentiment or emotion)")
    parser.add_argument("--benchmark_dataset_name", type=str, required=False, choices=["Jira", "StackOverflow", "CodeReview", "JavaLib"], help="name of benchmark dataset")
    parser.add_argument("--emotion_type", type=str.lower, required=False, default=None, choices=["anger", "love", "deva", "joy", "sad"], help="specify emotion dataset")
    
    parser.add_argument("--use_own_dataset", action='store_true', help="whether use your own dataset or not")
    parser.add_argument("--own_dataset_dir", type=str, required=False, default=None, help="directory of your train data file")
    parser.add_argument("--own_dataset_file", type=str, required=False, default=None, help="file name of your train data file")
    parser.add_argument("--sentisead", action='store_true', help="This will load code to run sentisead")

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

    try:
        # use provided dataset
        if not args.use_own_dataset:
            if args.benchmark_dataset_name is None:
                raise ValueError("should provide benchmark dataset name")

            if args.task == "sentiment":
                # data_path = "../../data/benchmark_dataset/sentiment/%s.txt" % args.benchmark_dataset_name
                data_path = "../../data/benchmark_dataset/sentiment/%s.tsv" % args.benchmark_dataset_name
                label2index_path = "label2index/sentiment/label2index_%s.json" % args.benchmark_dataset_name 

            else:
                trans_dict = {"Jira" : "JIRA", "StackOverflow" : "SO"}
                if args.benchmark_dataset_name not in trans_dict:
                    raise ValueError("invalid dataset name for emotion task")
                data_file_name = "%s_%s" % (trans_dict[args.benchmark_dataset_name ], args.emotion_type.upper())
                data_path = "../../data/benchmark_dataset/emotion/%s/%s.txt" % (args.benchmark_dataset_name , data_file_name)

                if args.emotion_type == 'deva':
                    if args.benchmark_dataset_name != "Jira":
                        raise ValueError("invalide dataset name for deva, requires Jira")
                    label2index_path = "label2index/emotion/label2index_5class.json" 
                else:
                    label2index_path = "label2index/emotion/label2index_2class.json"
                
            # load data and label2index file
            data_pair = load_data(data_path)

            with open(label2index_path, "r") as f_label:
                label2index = json.load(f_label)
            index2label = {i: l for l, i in label2index.items()}
        
        elif args.sentisead is not None:
            print("=============== We are going to train SentiMoji against Sentisead dataset ==============")
            label2index = {"0": 0, "1": 1, "-1": 2}
            index2label = {i: l for l, i in label2index.items()}

        # prepare your own data
        else:
            if args.own_dataset_dir is None or args.own_dataset_file is None:
                raise ValueError("should specify your own dataset directory and filename")

            # load data
            data_path = "{}/{}".format(args.own_dataset_dir, args.own_dataset_file)
            data_pair = load_data(data_path)

            # generate label2index file
            labels = set([pair[1] for pair in data_pair])
            label2index = {}
            for label in labels:
                label2index[label] = len(label2index)
            index2label = {i: l for l, i in label2index.items()}

            label2index_path = "{}/{}".format(args.own_dataset_dir, "label2index.json")
            with open(label2index_path, 'w') as f:
                json.dump(label2index, f)
    
    except RuntimeError as e:
        print("Error:", repr(e))

    # split 5 fold
    # data_5fold = prepare_5fold(data_pair)

    # sentence tokenizer (MAXLEN means the max length of input text)
    st = SentenceTokenizer(vocabulary, MAX_LEN)
    fold = 0

    # print(label2index)
    # 5 fold

        # dataset = dataset.lower()
    input_file =  os.path.join(base_dir, "data", "Disa_ResultsConsolidatedWithEnsembleAssessment.xlsx")
    
    datasets = ["DatasetLinJIRA", "BenchmarkUddinSO", "DatasetLinAppReviews", 
                "DatasetLinSO", "DatasetSenti4SDSO", "OrtuJIRA"]
    # datasets = [ "OrtuJIRA"]
    # dataset = "OrtuJIRA"
                # model 
    # model = deepmoji_architecture(nb_classes=nb_classes,
    #                             nb_tokens=nb_tokens,
    #                             maxlen=MAX_LEN, embed_dropout_rate=0.25, final_dropout_rate=0.5, embed_l2=1E-6)
    # # model.summary()

    # # load pretrained representation model
    # load_specific_weights(model, model_path, nb_tokens, MAX_LEN,
    #                     exclude_names=["softmax"])
    for dataset in datasets:
        dataset = dataset.lower()
        for fold in range(10):
        # for item in data_5fold:
            # prepare training, validation, testing set
            # train_pair, test_pair = get_train_test_data(infile=input_file, dataset = dataset, fold=fold)

            train_id, train_text, train_label, test_id, test_text, test_label = get_train_test(infile=input_file, dataset = dataset, fold=fold)
            # print(type(train_text[0]))
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
            # # model.summary()

            # # load pretrained representation model
            load_specific_weights(model, model_path, nb_tokens, MAX_LEN,
                                exclude_names=["softmax"])
            # 
            # # train model
            model, acc = finetune(model, [train_X, val_X, test_X], [train_y, val_y, test_y], nb_classes, 100,
                                  method="chain-thaw", verbose=2, nb_epochs=1)
            
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

            # precision = precision_score(test_y, pred_y, average=None)
            # recall = recall_score(test_y, pred_y, average=None)
            # f1score = f1_score(test_y, pred_y, average=None)
            labels = list(set(test_y))
            precision = precision_score(test_y, pred_y, average=None, labels = labels)
            recall = recall_score(test_y, pred_y, average=None, labels = labels)
            f1score = f1_score(test_y, pred_y, average=None, labels = labels)

            for index in range(0, len(labels)):
                print("label: %s" % index2label[index])
                print("Precision: %.3f, Recall: %.3f, F1 score: %.3f" % (precision[index], recall[index], f1score[index]))
            print("*****************************************")

            # save predict result
            if not args.use_own_dataset:
                if args.task == "sentiment":
                    save_name = "result_%s_%s_fold%d.txt" % (args.model, args.benchmark_dataset_name, fold)
                elif args.task == "emotion":
                    save_name = "result_%s_%s_%s_fold%d.txt" % (args.model, args.benchmark_dataset_name, args.emotion_type, fold)
            elif args.sentisead:
                save_name = dataset +"_result_fold%d.txt" % fold
                # os.path.join(dataset, save_name)
            else:
                save_name = "result_fold%d.txt" % fold        
            save_name = os.path.join(out_dir, save_name)
            # if(not os.path.exists(save_name)):
            #         os.makedirs(save_name)
            with open(save_name, "w", encoding="utf-8") as f:
                for i in range(0, len(test_text)):
                    f.write("%s\t%s\t%s\t%s\r\n" % (test_id[i], test_text[i], index2label[pred_y[i]], test_label[i]))
            print("#%d test results has been saved to: %s" % (len(test_text), save_name))

            fold += 1
            output_dir = "../../model/trained_model" + str(fold) + ".h5"
            if args.sentisead:
                output_dir = "../../model/sentisead/"
                output_dir = os.path.join(output_dir, dataset)
                if(not os.path.exists(output_dir)):
                    print("creating model file %s" % output_dir)
                    os.makedirs(output_dir)
                output_dir = os.path.join(output_dir, "trained_model" + str(fold) + ".h5" )
            # model.save_weights(output_dir)
            # print("Trained Models output has been saved to " + output_dir)

            # if(fold == 2):
            #     break # break
