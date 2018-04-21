fileID = fopen('exp-timebreakdown.txt','a');
fprintf(fileID,'-------------------------------------------------------------------------------------\n');        
fprintf(fileID,'Covtype Experiment\n');
fclose(fileID);
addpath('../libsvm-3.14-nobias/matlab');
maxNumCompThreads(4);
[y X] = libsvmread('../data/covtype.libsvm.binary.scale');
l = size(X,1);
p = randperm(l);
a = floor(l/5);
testX = X(p(1:a),:); testy = y(p(1:a));
trainX = X(p(a+1:end),:); trainy = y(p(a+1:end));

%% train/test rbf kernel SVM
ncluster = 64;
gamma = 32;
C = 32;
timebegin = cputime;
model = dcsvm_rbf_train(trainy, trainX, C, gamma, ncluster);
[labels accuracy] = dcsvm_test(testy, testX, model);
trainingtime_erl = cputime - timebegin;
fprintf('=============================================\n');
fprintf('DC Early RBF kernel, test accuracy %g training time \n', accuracy, trainingtime_erl);


%fprintf('Start training Gaussian kernel SVM\n');
%timebegin = cputime;
%model_exact = dcsvm_rbf_train_exact(trainy, trainX, C, gamma);
%trainingtime = cputime - timebegin;
%[labels_exact accuracy_exact] = dcsvm_test(testy, testX, model_exact);
%fprintf('=============================================\n');
%fprintf('DC Early RBF kernel, test accuracy %g training time \n', accuracy, trainingtime_erl);
%fprintf('=============================================\n');
%fprintf('RBF kernel, DC-SVM test accuracy %g, training time %g seconds\n', accuracy_exact, trainingtime);




%% WARNING: polynomial training is slow
%% train/test polynomial kernel SVM
%{
ncluster = 64;
gamma = 8;
degree = 2;
C = 2;
model1 = dcsvm_poly_train(trainy, trainX, C, gamma, degree, ncluster);
[labels1 accuracy1] = dcsvm_test(testy, testX, model1);
fprintf('polynomial kernel, test accuracy %g\n', accuracy1);
%}
