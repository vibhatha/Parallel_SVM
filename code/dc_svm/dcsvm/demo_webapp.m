addpath('../libsvm-3.14-nobias/matlab');
maxNumCompThreads(4);

[trainy, trainX] = libsvmread('../data/webspam_wc_normalized_unigram.svm');
[testy, testX] = libsvmread('../data/webspam_wc_normalized_unigram.svm');
trainy = double(trainy);
trainX = double(trainX);
testy = double(testy);
testX = double(testX);
%% train/test rbf kernel SVM
ncluster = 10;
gamma = 2;
C = 32;
fprintf('Start training Gaussian kernel SVM with early prediction\n', ncluster);
timebegin = cputime;
model = dcsvm_rbf_train(trainy, trainX, C, gamma, ncluster);
trainingtimeerl = cputime - timebegin;
[labels accuracy] = dcsvm_test(testy, testX, model);


fprintf('Start training Gaussian kernel SVM\n');
timebegin = cputime;
model_exact = dcsvm_rbf_train_exact(trainy, trainX, C, gamma);
trainingtime = cputime - timebegin;
[labels_exact accuracy_exact] = dcsvm_test(testy, testX, model_exact);
fprintf('=============================================== \n');
fprintf('RBF kernel, DCSVM-early test accuracy %g, training time %g seconds\n', accuracy, trainingtimeerl);
fprintf('=============================================== \n');
fprintf('RBF kernel, DC-SVM test accuracy %g, training time %g seconds\n', accuracy_exact, trainingtime);

%{
%% train/test polynomial kernel SVM
ncluster = 10;
gamma = 32;
degree = 2;
C = 0.125;
fprintf('Start training polynomial kernel SVM with early prediction\n');
timebegin = cputime;
model1 = dcsvm_poly_train(trainy, trainX, C, gamma, degree, ncluster);
trainingtime = cputime - timebegin;
[labels1 accuracy1] = dcsvm_test(testy, testX, model1);
fprintf('polynomial kernel, DC-SVM early test accuracy %g, training time %g seconds\n', accuracy1, trainingtime);

fprintf('Start training polynomial kernel SVM\n');
timebegin = cputime;
model1_exact = dcsvm_poly_train_exact(trainy, trainX, C, gamma, degree);
trainingtime = cputime - timebegin;
[labels1_exact accuracy1_exact] = dcsvm_test(testy, testX, model1_exact);
fprintf('polynomial kernel, DC-SVM test accuracy %g, training time %g seconds\n', accuracy1_exact, trainingtime);
%}
