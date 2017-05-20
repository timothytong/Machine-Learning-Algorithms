function [ y_est ] =  my_knn(X_train,  y_train, X_test, k, type)
%MY_KNN Implementation of the k-nearest neighbor algorithm
%   for classification.
%
%   input -----------------------------------------------------------------
%   
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {0,1} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o k        : number of 'k' nearest neighbors
%       o type   : (string), type of distance {'L1','L2','LInf'}
%
%   output ----------------------------------------------------------------
%
%       o y_est   : (1 x M_test), a vector with estimated labels y \in {0,1} 
%                   corresponding to X_test.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~, M_train] = size(X_train);
M_test = size(X_test, 2);
D = zeros(1, M_train);
y_est = zeros(1, M_test);
 
for i = 1:M_test
    for j = 1:M_train
        % compute the pairwise distances between test points and training points
        D(j) = my_distance(X_test(:,i), X_train(:,j), type);
    end
    
    % sort
    [~, idx] = sort(D, 'ascend');
        
    % extract points with corresponding indices
    k_idx = idx(1:k);
    y_k = y_train(k_idx);

    % majority vote
    y_est(i) = sum(y_k) > k/2;
end

end