dataset
============

https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
juliet:/share/jproject/fg474/dataset/libsvm_data

name    |source  type   | class |  training size  | testing size  |  feature
--------| -----------   | ----  | -------------   | ------------  | ---------
ijcnn1 | [DP01a] | classification | 2 |  49,990 | 91,701|  22
covtype| UCI | classification | 7 |  581,012|  |   54
webspam| Webb Spam Corpus [ST06a]|    classification | 2 |  350,000 |    16,609,143
mnist8m| Invariant SVM [GL07b]  | classification | 10 | 8,100,000   |    784

training/test split

dataset | number of training samples | number of testing samples  |  notes
------- | ------------------------   | ----------------------     | ---------
ijcnn1  |  49,990 | 91,701  |  ijcnn1.tr + ijcnn1.val as training set
covtype | 464,810 | 116,202 | python datasplit.py covtype.scale01
webspam | 280,000 | 70,000  | python datasplit.py webspam_wc_normalized_unigram.svm
mnist8m | 8,000,000 | 100,000 | python datasplit.py mnist8m.scale 100000 

