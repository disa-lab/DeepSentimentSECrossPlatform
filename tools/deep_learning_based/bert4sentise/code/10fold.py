from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import warnings
warnings.filterwarnings('ignore')

import csv
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import shutil
import run_sosc
import argparse
from pathlib import Path
import shutil
from run_sosc import *
import pandas as pd

def disa_create_train_test_set(input_file, dataset, fold):
    dataset = dataset.lower()
    # input_file =  os.path.join(dataset_dir, "ResultsConsolidatedWithEnsembleAssessment.xlsx")
    # df_all = pd.read_excel(input_file, sheet_name="Sheet1", usecols="S, AF, T, AX", names=['dataset', 'oracle', 'text', 'id'])
    df_all = pd.read_csv(input_file, skiprows=1, names=['id', 'dataset', 'text', 'oracle'], dtype={'oracle': str})
    # df_all.insert(loc=0, column="id", value=df_all.index + 1)
    # df_all['id'] = df_all.index
    df_all = df_all[['id', 'text', 'oracle', 'dataset']]
    # print("length of all datasets %d" % len(df_all))

    df_all.loc[df_all.oracle == '0', 'oracle'] = 'Neutral'
    df_all.loc[df_all.oracle == '-1', 'oracle'] = 'Negative'
    df_all.loc[df_all.oracle == '1', 'oracle'] = 'Positive'
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


    dataset_dir =  "/home/mdabdullahal.alamin/alamin/sentiment/bert/dataset/"
    train_df.to_csv( dataset_dir + "train.tsv", sep='\t', index=False, header = None)
    test_df.to_csv( dataset_dir + "test.tsv", sep='\t', index=False, header = None)



def disa_create_cross_platform_train_test_set(input_file, train_dataset):
    train_dataset = train_dataset.lower()
    # test_dataset = test_dataset.lower()
    df_all = pd.read_csv(input_file, skiprows=1, names=['id', 'dataset', 'text', 'oracle'], dtype={'oracle': str})
    df_all = df_all[['id', 'text', 'oracle', 'dataset']]


    df_all.loc[df_all.oracle == '0', 'oracle'] = 'Neutral'
    df_all.loc[df_all.oracle == '-1', 'oracle'] = 'Negative'
    df_all.loc[df_all.oracle == '1', 'oracle'] = 'Positive'


    train_df = df_all[df_all['dataset'].astype(str).str.lower().str.contains(train_dataset)]
    train_ids = train_df['id'].tolist()
    test_df = df_all[~df_all['id'].isin(train_ids)]



    # test_df = df_all[df_all['dataset'].astype(str).str.lower().str.contains(test_dataset)]

    train_df = train_df.drop('dataset', axis = 1) # 0 means rows 1 means column
    test_df = test_df.drop('dataset', axis = 1) # 0 means rows 1 means column

    print("Train dataset: %s and len %d " % ( train_dataset, len(train_df)))
    print("Test dataset: len %d " % ( len(test_df)))
    assert len(train_df) + len(test_df) == len(df_all)


    dataset_dir =  "/home/mdabdullahal.alamin/alamin/sentiment/bert/dataset/"
    train_df.to_csv( dataset_dir + "train.tsv", sep='\t', index=False, header = None)
    test_df.to_csv( dataset_dir + "test.tsv", sep='\t', index=False, header = None)



if __name__ == "__main__":
    ######Configuration for running BERT 10 fold ########
    print("argument parser commented out")
    tf.get_logger().setLevel(logging.ERROR)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--data_dir', default='datasets', help='Directory for the dataset')
    # parser.add_argument('-o', '--out_dir', default='out', help='Directory for the output')
    # args = parser.parse_args()

    root_dir = os.getcwd()
    ###Temporary directory for saving BERT results###
    source_dir = os.path.join(root_dir, 'sosc_output')
    # out_dir = os.path.join(root_dir, args.out_dir)
    # out_dir = args.out_dir # alamin_edit
    out_dir = "/home/mdabdullahal.alamin/alamin/sentiment/bert/output/out/"
    num_of_folds = 10
    undersample = 'no' #provide yes or no
    # data_dir = os.path.join(root_dir, args.data_dir)
    # data_dir =args.data_dir # alamin_edit
    data_dir = "/home/mdabdullahal.alamin/alamin/sentiment/bert/dataset"
    datasets = ["DatasetSenti4SDSO", "OrtuJIRA", "GitHub"]
    # input_file = os.path.join(data_dir, "combined.csv")
    input_file = "/home/mdabdullahal.alamin/alamin/sentiment/cross_platform/dataset/processed/combined.csv"

    test_type = "inner_platform" # test_type can be inner_platform or cross platform
    test_type = "cross_platform"
    test_type = "both"


    if(test_type == "cross_platform" or test_type=="both"):
        print("Going to test in cross platform settings")
        for train_dataset in datasets:
            train_dataset = train_dataset.lower()            
            print("train dataset is %s " % (train_dataset))

            dest_file_name = 'bert_train_' + train_dataset + ".csv"
            dest_file = os.path.join(out_dir, dest_file_name)
            des_dir = out_dir
            if(not os.path.exists(des_dir)):
                # # print(des_dir)
                # shutil.rmtree(des_dir)
                # print("==========> Duplicate directory is removed %s " % (des_dir))
                os.makedirs(des_dir)
            # create_trn_dev_test_set(data_dir, fold, num_of_folds, batches, train_batches)
            disa_create_cross_platform_train_test_set(input_file=input_file, train_dataset=train_dataset)
            run_sosc.main()
            # dest = shutil.move(source_dir, des_dir, copy_function = shutil.copytree)
            shutil.copy2('sosc_output/test_results.csv', dest_file)
            print("=========> Result has been copied to %s" % (dest_file))
            shutil.rmtree("sosc_output/", ignore_errors=True)
            # break


    if(test_type == "inner_platform" or test_type=="both"):
        for dataset in datasets:
            dataset = dataset.lower()            
            for fold in range(num_of_folds):
                print("\n ==================> Dataset %s and Fold: %s "% (dataset, str(fold)))
                dest_file_name = 'bert_' + dataset + "_" + str(fold) + ".csv"                
                des_dir = os.path.join(out_dir, "inner")
                dest_file = os.path.join(des_dir, dest_file_name)
                if(not os.path.exists(des_dir)):
                    os.makedirs(des_dir)
                disa_create_train_test_set(input_file=input_file, dataset=dataset, fold=fold)
                run_sosc.main()
                # dest = shutil.move(source_dir, des_dir, copy_function = shutil.copytree)
                shutil.copy2('sosc_output/test_results.csv', dest_file)
                print("=========> Result has been copied to %s" % (dest_file))
                shutil.rmtree("sosc_output/", ignore_errors=True)
            #     break
            # break







