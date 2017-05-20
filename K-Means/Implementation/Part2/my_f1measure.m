function [f1measure] =  my_f1measure(cluster_labels, class_labels)
%MY_F1MEASURE Computes the f1-measure for semi-supervised clustering
%
%   input -----------------------------------------------------------------
%   
%       o class_labels     : (M x 1),  M-dimensional vector with true class
%                                       labels for each data point
%       o cluster_labels    : (M x 1),  M-dimensional vector with predicted 
%                                       cluster labels for each data point
%   output ----------------------------------------------------------------
%
%       o f1_measure      : f1-measure for the clustered labels
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = length(class_labels);
true_K = unique(class_labels);
found_K = unique(cluster_labels);

nClasses = length(true_K);
nClusters = length(found_K);

% Initializing variables

C = zeros(1, nClasses);
F1 = zeros(nClusters, nClasses);

for ii = 1:nClasses
    C(ii) = sum(class_labels == ii);
    
    for jj = 1:nClusters
        n_ik = sum((class_labels == ii) .* (cluster_labels == jj));
        k = sum(cluster_labels == jj);    
        
        % Implement the precision equation here
        % Precision: proportion of datapoints of same class in cluster
        p = n_ik / k;
        
        % Implement the recall equation here
        % Recall: proportion of datapoints correctly classified/clusterized
        r = n_ik / C(ii);

        % Implement the F1 measure for each cluster here
        F1(jj, ii) = 2*r*p/(r+p);
    end
end

% Implement the F1 measure for all clusters here
f1measure = sum((C/M) .* max(F1));

end
