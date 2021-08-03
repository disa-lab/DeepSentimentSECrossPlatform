import os
import numpy as np
import pandas as pd 
import sklearn.metrics
import argparse
import csv

parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', type=str,  help='')
parser.add_argument('--answer_path', type=str,  help='')
parser.add_argument('--task', type=str,  default="sosc", help='default:sosc')
args = parser.parse_args()

test = args.answer_path[:-4] + '_wp' + args.answer_path[-4:]
test2 = args.output_path[:-4] + '_wr' + args.output_path[-4:]

testdf = pd.read_csv(args.answer_path, sep="\t", header=None)
preddf = pd.read_csv(args.output_path, sep="\t", header=None)

if args.task == "sosc":
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    preds = [np.argmax(v) for v in pred]

    label_ids = []
    str_to_int_mapper = {'Negative':0,'Neutral':1,'Positive':2}
    int_to_str_mapper = {0:'Negative',1:'Neutral',2:'Positive'}
    for index, row in testdf.iterrows():
        label_ids.append(str_to_int_mapper[row[2]])

    p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_true=label_ids, y_pred=preds)
    ac = sklearn.metrics.accuracy_score(y_true=label_ids, y_pred=preds)
    results = dict()
    results["f1 score"] = f
    results["recall"] = r
    results["precision"] = p
    results["accuracy"] = ac
    results["support"] = s

    '''
    cm = sklearn.metrics.confusion_matrix(y_true=label_ids, y_pred=preds)

    with open(test, 'w') as wf:
        with open(args.answer_path, 'r') as rf:
            tsv_writer = csv.writer(wf, delimiter="\t")
            reader = csv.reader(rf, delimiter="\t", quotechar=None)
            for i,line in enumerate(reader):
                tsv_writer.writerow([line[1], line[2], int_to_str_mapper[preds[i]]])

    '''

for k,v in results.items():
    print("{:11s} : {:.4%} {:.4%} {:.4%} {:.4%}".format(k,v[0],v[1],v[2],v[3]))

'''
with open(test2, 'wt') as tf:
    tsv_writer = csv.writer(tf, delimiter='\t')
    tsv_writer.writerow(["Metric","Negative","Neutral","Positive"])
    for k,v in results.items():
        tsv_writer.writerow([k,v[0],v[1],v[2]])

    tsv_writer.writerow(["CM","Negative","Neutral","Positive"])
    label_list = ['Negative','Neutral','Positive']
    for i,j in enumerate(cm):
        tsv_writer.writerow([label_list[i],j[0],j[1],j[2]])
'''
