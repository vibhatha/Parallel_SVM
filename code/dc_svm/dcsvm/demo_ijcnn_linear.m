fileID = fopen('exp-timebreakdown.txt','a');
fprintf(fileID,'-------------------------------------------------------------------------------------\n');        
fprintf(fileID,'IJCNN Experiment\n');
fclose(fileID);

addpath('../libsvm-3.14-nobias/matlab');
maxNumCompThreads(1);

[trainy, trainX] = libsvmread('../data/ijcnn1.train');
[testy, testX] = libsvmread('../data/ijcnn1.t');
trainy = double(trainy);
trainX = double(trainX);
testy = double(testy);
testX = double(testX);
%% train/test rbf kernel SVM
ncluster = 10;
gamma = 1;
C = 7;
fprintf('Start training Linear kernel SVM with early prediction\n', ncluster);
timebegin = cputime;
model = dcsvm_poly_train(trainy, trainX, C, gamma, 1,ncluster);
trainingtimeerl = cputime - timebegin;
[labels accuracy] = dcsvm_test(testy, testX, model);


fprintf('Start training Linear kernel SVM\n');
timebegin = cputime;
model_exact = dcsvm_poly_train_exact(trainy, trainX, C, gamma,1);
trainingtime = cputime - timebegin;
[labels_exact accuracy_exact] = dcsvm_test(testy, testX, model_exact);
fprintf('=============================================== \n');
fprintf('Linear kernel, DCSVM-early test accuracy %g, training time %g seconds\n', accuracy, trainingtimeerl);
fprintf('=============================================== \n');
fprintf('Linear kernel, DC-SVM test accuracy %g, training time %g seconds\n', accuracy_exact, trainingtime);
fileID = fopen('exp-timebreakdown.txt','a');
fprintf(fileID,'Earl Acc: %f, Earl Time: %f || Exact Acc: %f, Exact Time: %f \n', accuracy, trainingtimeerl, accuracy_exact, trainingtime);
fprintf(fileID,'-------------------------------------------------------------------------------------\n');        
fclose(fileID);

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
