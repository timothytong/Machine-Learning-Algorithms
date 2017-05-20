%%  Test Implementation of Principal Component Analysis (PCA)
%    on 9D Dataset. 
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
%% --> Fill in ML_toolbox Path HERE!!! --> %%
clear all;
close all;
clc;
ml_toolbox_path = 'Write your ML_toolbox-master path here';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               Load 9D PCA Testing Dataset                  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Breast-Cancer-Wisconsin Dataset 
% https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
% Nine real-valued features are computed for each cell nucleus: 
%   Feature                          Range
%   1. Clump Thickness               1 - 10
%   2. Uniformity of Cell Size       1 - 10
%   3. Uniformity of Cell Shape      1 - 10
%   4. Marginal Adhesion             1 - 10
%   5. Single Epithelial Cell Size   1 - 10
%   6. Bare Nuclei                   1 - 10
%   7. Bland Chromatin               1 - 10
%   8. Normal Nucleoli               1 - 10
%   9. Mitoses                       1 - 10


[X,labels,class_names] = ml_load_data(strcat(ml_toolbox_path,'data/breast-cancer-wisconsin.csv'),'csv','last');

options.labels      = labels;
options.class_names = {'Benign','Malignant'};
options.title       = 'Breast-Cancer-Wisconsin (Diagnostic) Dataset';
h0 = ml_plot_data(X,options);
axis equal;

% Transpose data to have columns as datapoints
X = X'; labels = labels';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             Test PCA Projection with your functions        %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Principal Directions and Components
[V, L, Mu] = my_pca(X);

% Project Data to Choosen Principal Components
p = 9;
[A_p, Y] = project_pca(X, Mu, V, p);

% Visualize Projected Data
plot_options             = [];
plot_options.is_eig      = true;
plot_options.labels      = labels;
plot_options.class_names = {'Benign','Malignant'};
plot_options.title       = 'Projected Breast-Cancer-Wisconsin data with PCA';

h1 = ml_plot_data(Y',plot_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Finding the Optimal P by Analyzing Eigenvalues     %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot EigenValues to try to find the optimal "p"
plot_eigenvalues(L);

% Project Data to Choosen Principal Components
p = 3;
[A_p, Y] = project_pca(X, Mu, V, p);

% Visualize Projected Data
plot_options             = [];
plot_options.is_eig      = false;
plot_options.labels      = labels;
plot_options.plot_labels = {'eig 1','eig 2','eig 3'};
plot_options.class_names = {'Benign','Malignant'};
plot_options.title       = 'Reduced Breast-Cancer-Wisconsin Dataset';

h2 = ml_plot_data(Y',plot_options);
axis equal

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Finding the Optimal P wrt. Desired Explained Variance     %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find 'p' which yields the desired 'Explained Variance'
Var = 0.9;
[ p ] = explained_variance( L, Var );

% Project Data to Choosen Principal Components
[A_p, Y] = project_pca(X, Mu, V, p);

% Visualize Projected Data
plot_options             = [];
plot_options.is_eig      = true;
plot_options.labels      = labels;
plot_options.class_names = {'Benign','Malignant'};
plot_options.title       = 'Reduced Breast-Cancer-Wisconsin Dataset';

h3 = ml_plot_data(Y',plot_options);
axis equal
