# Deep Sentiment SE Cross-Platform

This repository contains the data, code, pre-trained models, experiment results for our ressearch project - **An Empirical Study of Three Deep Learning Sentiment Detection Tools for SE in Cross-Platform Settings** 

## Overview of Sentiment4SE
A deep learning based Sentiment Analysis tool for Software Engineering datasets.

In this research work we studied the performance of different sentiment analysis tools for Software Engineering datasets. We specifically focused on the performance
of newly developed deep learning based tools such as (BERT4SentiSE, SEntiMoji, RNN4SentiSE).

## Benchmark Datasets
For this study, we used three benchmark cross platform datasets from 
<!-- 1. **GitHub**, 
2. **Jira**, and 
3. **Stack Overflow**.  -->
- The **[GitHub dataset](https://dl.acm.org/doi/abs/10.1145/3379597.3387446?casa_token=IVm2ckwP7tkAAAAA%3AiI5wI10i1PLqO39hFeWZgN1PcXNrDOUO61cbVuglZcAAm9uY9WkWngpiN9fmPsrhNb5FVasPGjDPtg)** contains around 7000 pull requestand commit comments. The  dataset is well balanced of 28% positive, 29% negative, and 43% neutral emotions.

- The **[Jira dataset]()** contains around 6000 issue commentsand sentences of open source software projects (eg. Apache, Spring) annotated by software developers. This dataset was originally labelled with six emotions (i.e. love, joy, surprise, anger, fear, and sadness). In order to be consistent with other datasets, we can translate **love** and **joy** as a **positive emotion**, **anger** and **sadness** as a **negative emotion**. **surprise** cases are discarded as they could be either **positive** or **negative**.  Finally, the absence of emotions is labelled as **neutral**.  The  dataset is  not  well  balanced and the ratio is  **19%**  positive**,  **14%** negative, and **67%** neutral emotions.

- The **[Stack Overflow dataset](https://github.com/collab-uniba/Senti4SD)** contains  4423  Stackoverflow  posts  including  questions,  comments  and  answers, manually  annotated  by  twelve  trained  codes.  Each  post  was annotated by three raters and received the final polarity based on  majority  voting.  The  dataset  is  quite  well  balanced of  **35%** positive,  **27%**  negative, and **38%**  neutral  emotions.  

## Sentiment Analysis Tools
For this study, we used three deep learning based tools and two shallow supervised ML tools.

- Deep learning based tools:
  1. **[SEntiMoji](https://dl.acm.org/doi/abs/10.1145/3338906.3338977?casa_token=JvLUZ9UaM-MAAAAA%3AaX4Xio8roPclBdjeTgfKQ0pHqCr4vZxo3lMxcSW6SbWIYkxba6hjc9534BRBfZqaz09xUuFGzf869A)**: proposed by Chen et al. in 2019, is an  SE customized  sentiment  classifier  based  on  **[DeepMoji](https://arxiv.org/abs/1708.00524)**. It learns vector representation of texts by leveraging how emojis  are  used  and  other  texts  in  Tweeter  and  GitHub. The authors reported that it outperforms existing methodsby **[Lin et al.’s](https://sentiment-se.github.io/replication.zip)** dataset.
  
  2. **[BERT4SentiSE](https://ieeexplore.ieee.org/document/9240599)**: proposed by Biswas et al. in 2020, is a BERT based pre-trained transformer model. This  explores  the effectiveness  of  using  **[BERT](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ)**  based  model  for  SA4SE.  The  authors  report  that the BERT classifier achieves reliable performance for SE datasets. They also report BERT combined with a larger dataset provides a better result.

  3. **[RNN4SentiSE](https://2019.msrconf.org/details/msr-2019-papers/22/Exploring-Word-Embedding-Techniques-to-Improve-Sentiment-Analysis-of-Software-Enginee)**: proposed by Biswas et al. in 2019,  is  based  on  generic  word embedding  from  Google  news  data.  Its  generic  word embedding  was  also  updated  using  software  domain-specific word embedding from stack overflow posts. For this study’s purpose, we  used this tool as a base model for  RNN  based  SA4SE  tools  that  do  not  use  any  pre-trained  word  vectors  and  fine-tuning.  So,  for  this  tool, we generated our own word embedding.

- Shallow ML based tools:
  1. **[Senti4SD](https://link.springer.com/article/10.1007/s10664-017-9546-9)**: proposed by Calefato et al. in 2018,  is  a  polarity  classifier  that uses  a  bag  of  words  (BoW),  sentiment  lexicons,  word  em-bedding  as  features.  This  toolkit  was  originally  trained  and validated on 4K questions and answers of Stack Overflow. This toolkit allows customization by using gold standard dataset as input.
   
  2. **[SentiCR](https://ieeexplore.ieee.org/document/8115623)** proposed by Ahmed et al. in 2017,  extracts  term  frequency  in-verse  document  frequency  for  Bag  of  words  and  uses  it  as a  feature.  It  implements  pre-processing  of  input  and  handles negations, stop-words, removes code snippets. It also leverages SMOTE  to  handle  class  imbalance  in  the  training  dataset.  It is based on Gradient Boosting Tree (GBT) and it is originally trained and tested on 2000 code-review comments.

- Rule based tools:
  1. **[SentistrengthSE](https://doi.org/10.1016/j.jss.2018.08.030)**: is developed by Md RakibulIslam and Minhaz F.Zibran on top of [SentiStrength](https://doi.org/10.1002/asi.21416) by introducing rules and sentiment words specific to Software Engineering. 
  

<!-- ## Tools Overview
Tools folder contain their own `readme.md` file where it contains information about the tool from the original repository as well as some of our updated documentation. For each tools we provide the `requirements.txt` file that contains that environment configuration that we used during our experiments. -->

## Repo Structure
- `/analysis` dir
  - Contains the Jupyter notebooks that were used during analyzing the outputs and producing the results.
- `/datasets` dir
  - Contains the raw and processed datasets.
  - `/combined.csv` file 
    - contains the combined datasets that is used by the following deep and shallow machine learning based sentiment analysis tools. 
    - contains 10-fold **stratified** sampling with Scikit-learn. This 10-fold sampling is used to report the performance of echo tools in within-platform settings.

- `/generated_output` dir
  - Contains generated combined outputs from all the tools based on all the datasets

- `/manual_labeling` dir
  - Contains the files that we generated after performing manual labeling.
  - `error_categorization.csv` file contains labeling of BERT4SentiSE and SEntiMoji errors.
  - `sentistrengthse_errors.csv` file contains labeling of SentistrentSE errors.
- `/tools` dir
  - Contains source codes of all the sentiment analysis tools that are used in this study
  - `/deep_learning_based` dir
    - `/bert4sentise` contains replication package of **[BERT4SentiSE](https://www.dropbox.com/sh/0dzw55rqo7e6k2g/AADS5M6QIbi9w3ntKqVesWtWa/Code?dl=0&subfolder_nav_tracking=1)**
    - `/rnn4sentise` contains the complete replication package of **[RNN4SentiSE](https://www.dropbox.com/sh/0dzw55rqo7e6k2g/AADS5M6QIbi9w3ntKqVesWtWa/Code?dl=0&subfolder_nav_tracking=1)**
    - `/sentimoji` contains the complete replication package of **[SEntiMoji](https://github.com/SEntiMoji/SEntiMoji)**
  
  - `/shallow_learning_based` dir
    - `/senti4sd` contains the complete replication package of **[Senti4SD](https://github.com/collab-uniba/Senti4SD)**
    - `/senticr` contains the complete replication package of **[SentiCR](https://github.com/senticr/SentiCR/)**


### Note
- Tools folder contain their own `readme.md` files where it contains information about the tool from the original repository as well as some of our updated documentation. For each tools we provide the `requirements.txt` file that contains that environment configuration that we used during our experiments.



## Declaration
We upload all the benchmark datasets that we used for this study to this repository for convenience.  We do not claim any rights on them because they were not generated and released by us,. If you use any of them, please make sure you fulfill the licenses that they were released with and consider citing the original papers. The folders in this repository contains links and licence information about the orginal repository.

