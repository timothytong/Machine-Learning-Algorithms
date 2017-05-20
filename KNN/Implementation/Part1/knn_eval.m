function [ ] = knn_eval( X_train, y_train, X_test, y_test, k_range )
%KNN_EVAL Implementation of kNN evaluation.
%
%   input -----------------------------------------------------------------
%   
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {0,1} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_test   : (1 x M_test), a vector with labels y \in {0,1} corresponding to X_test.
%       o k_range  : (1 X K), Range of k-values to evaluate
%
%   output ----------------------------------------------------------------
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

n_k = size(k_range, 2);
acc = zeros(1, n_k);
for i = 1:n_k
    y_est = my_knn(X_train, y_train, X_test, k_range(i), 'L2');
    acc(i) = my_accuracy(y_test, y_est);
end

plot(k_range, acc, '--or', 'LineWidth', 1);
ylabel('Acc')
xlabel('k')
title('Classification Evaluation for KNN')
grid on

end

