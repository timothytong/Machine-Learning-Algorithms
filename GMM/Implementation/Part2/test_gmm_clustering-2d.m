%%  Test Implementation of clustering using GMM
%    on a 2D Dataset. 
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             1) Load 2D GMM Testing Dataset                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1a) Load 2D dataset for testing and coparing with 
clear all; close all; clc;

load('../../TP4-GMM-Datasets/2D-GMM.mat')

% Visualize Dataset
options.labels      = labels;
options.class_names = {};
options.title       = '2D GMM Dataset';

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
colors     = hsv(4);
ml_plot_centroid(gmm.Mu',colors);hold on; 
ml_plot_gmm_contour(gca,gmm.Priors,gmm.Mu,gmm.Sigma,colors);
grid on; box on;
axis equal;

%%             1.b) Load 2D GMM Testing Dataset                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load 2D dataset for testing and coparing with 
clear all; close all; clc;

load('../../TP4-GMM-Datasets/2D-GMM-clustering.mat')

% Visualize Dataset
options.labels      = labels;
options.class_names = {};
options.title       = '2D GMM Dataset';

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
colors     = hsv(4);
ml_plot_centroid(gmm.Mu',colors);hold on; 
ml_plot_gmm_contour(gca,gmm.Priors,gmm.Mu,gmm.Sigma,colors);
grid on; box on;
axis equal;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              2) Finding the GMM model                      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% K-means Evaluation Parameters
K_range=1:10; cov_type = 'full'; repeats = 1;

% Evaluate gmm-em to find the optimal k
clc;
tic
gmm_eval(X, K_range, repeats, cov_type);
toc;

%% Pick the best K and view the GMM-PDF

K = 2; cov_type = 'full';  plot_iter = 0;
[Priors, Mu, Sigma] = my_gmmEM(X, K, cov_type,  plot_iter);

close all;
ml_plot_gmm_pdf(X, Priors, Mu, Sigma)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%            3) Cluster the data using the model             %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

type = 'soft';
softThresholds = [0.3, 0.7];
labels_gmm = gmm_cluster(X, Priors, Mu, Sigma, type, softThresholds);

% Plot the results
close all;
options.labels      = labels_gmm;
options.class_names = {};
options.title       = 'Data clustered with GMM';

if(strcmp(type,'soft'))
    options.colors(1,:) = 0.5*[1;1;1];
    options.colors(2:K+1,:) = hsv(K);
elseif (strcmp(type,'hard'))
    options.colors = hsv(K);
end

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
if(strcmp(type,'soft'))
    legend('Not clustered', 'Cluster 1', 'Cluster 2', 'Cluster 3');
elseif (strcmp(type,'hard'))
    legend('Cluster 1', 'Cluster 2', 'Cluster 3');
end

% Plot the cluster boundary

options.method_name = 'gmm';
options.class_names = {};
options.title       = 'Cluster boundaries';
options.labels = labels_gmm;
options.b_plot_boundary = true;

options.gmm.Mu         = Mu;
options.gmm.Priors     = Priors;
options.gmm.Sigma      = Sigma;
options.type           = type;
options.softThresholds = softThresholds;

if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = ml_plot_class_boundary(X',options);hold on;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%            4) Cluster the data using K-Means               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Find best K fror K-means Evaluation Parameters
% K-means Evaluation Parameters
K_range=1:10; type='L2'; repeats = 10;
init='random'; MaxIter = 100;

% Evaluate k-means to find the optimal k
clc;
kmeans_eval(X, K_range, repeats, init, type, MaxIter);

%% Select the best K and compute K-means 
best_k = 2;
[labels_kmeans, Mu] =  my_kmeans(X, best_k, init, type, MaxIter, 0);

%% Plot decision boundary
my_kmeans_result.distance    = type;
my_kmeans_result.K           = K;
my_kmeans_result.method_name = 'kmeans';
my_kmeans_result.labels      = labels_kmeans';
my_kmeans_result.centroids   = Mu';
my_kmeans_result.title       = sprintf('. My K-means result. k = %d, dist = %s',K, type);

if exist('hd','var') && isvalid(hd), delete(hd);end
hd = ml_plot_class_boundary(X',my_kmeans_result);
