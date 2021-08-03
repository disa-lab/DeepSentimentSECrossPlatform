import subprocess
import os

# output_dir =  "/home/mdabdullahal.alamin/alamin/sentiment/senti4sd/py_senti4sd/train_test_dataset/inner/"
datasets = [ "DatasetSenti4SDSO", "GitHub", "OrtuJIRA"]


type = "inner"
# type = "cross"
type = "both"

if(type == "cross" or type =="both"):
    for train_dataset in datasets:
        print("#" * 20)
        train_dataset = train_dataset.lower()
        for test_dataset in datasets:
            test_dataset = test_dataset.lower()
            if(train_dataset == test_dataset):
                continue

            train_file = os.path.join("train_test_dataset", train_dataset + ".csv")
            test_file = os.path.join("train_test_dataset", test_dataset + ".csv")
            output_file = "senti4sd_train_" + train_dataset + "_test_" + test_dataset + ".csv"
            # output_file = os.path.join("cross", output_file_name)
            print("train_file %s test_file %s and output_file: %s" % (train_file, test_file, output_file))



            # Going to train the dataset
            #  sh train.sh -i train_test_dataset/github.csv -i train_test_dataset/ortujira.csv -d c -j 32 -m DISA
            command = "sh train.sh -i %s -i %s -d c -j 32 -c 15000 -m DISA" % (train_file, test_file)
            print("Going to execute the following arguments %s" % (command))
            os.system(command)

            # Going to test the dataset
            # sh classification.sh -i train_file -j 32 -m "alamin_g_j" -o output_file
            command = "sh classification.sh -i %s -j 32 -c 15000 -m DISA.model -o %s" % (test_file, output_file)     
            print("Going to execute the following arguments %s" % (command))
            os.system(command)
            print("+" * 10)
        #     break
        # break


if(type == "inner" or type =="both"):
    for dataset in datasets:
        print("#" * 20)
        dataset = dataset.lower()
        for fold in range(10):
            train_file = os.path.join("train_test_dataset", "inner", dataset + "_train_" + str(fold) + ".csv")
            test_file = os.path.join("train_test_dataset", "inner", dataset + "_test_" + str(fold) + ".csv")
            output_file =  dataset + "_test_" + str(fold) + ".csv"

            print("train_file %s test_file %s and output_file: %s" % (train_file, test_file, output_file))


            # Going to train the dataset
            #  sh train.sh -i train_test_dataset/github.csv -i train_test_dataset/ortujira.csv -d c -j 32 -m DISA
            command = "sh train.sh -i %s -i %s -d c -j 32 -c 15000 -m DISA" % (train_file, test_file)
            print("Going to execute the following arguments %s" % (command))
            os.system(command)

            # Going to test the dataset
            # sh classification.sh -i train_file -j 32 -m "alamin_g_j" -o output_file
            command = "sh classification.sh -i %s -j 32 -c 15000 -m DISA.model -o %s" % (test_file, output_file)     
            print("Going to execute the following arguments %s" % (command))
            os.system(command)

            print("+" * 10)
        print("FOld loop over")
