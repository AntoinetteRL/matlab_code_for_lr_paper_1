clear
global_timer = tic;

%% Generate Signal
mortality_data = Lung_Cancer_Data_Processing();

mortality_data_norm = zscore(mortality_data,1,'all');

N = size(mortality_data,1); % Number of nodes
M = 3;  % Max cardinality

%% Set Other Constants
[H_cancer,C_cancer,avg_data_cancer] = KNN_Ground_Truth(mortality_data_norm, M);

%% Unique list of hyperedge possibilities
all_hyperedge_possibilities = false;  % Selecting whether to use every hyperedge possibility or a K-NN shortened list
neighbors = [0,15];  % Selects number of neighbors per cardinality if all_hyperedge_possibilities is false
if all_hyperedge_possibilities
    count = 1:N;
    Da = nchoosek(N,M);  % Number of potential hyperedges
    edge_cell = num2cell(nchoosek(count,M),2);
    listUnique = edge_cell;  % Cell array of all hyperedge permutations
else
    listUnique = generate_knn_hyperedges(mortality_data_norm,neighbors);
    Da = size(listUnique,1);
end

%% Distance Vector
smooth_type = 4;
switch smooth_type
    case 1
        z = smooth_square_sum(mortality_data_norm,listUnique);  % Distance vector squared sum
    case 2
        z = smooth_abs_sum(mortality_data_norm,listUnique);  % Distance vector absolute sum
    case 3
        z = smooth_abs_max(mortality_data_norm,listUnique);  % Distance vector absolute maximum
    case 4
        z = smooth_square_max(mortality_data_norm,listUnique);  % Distance vector squared maximum
end

%% S Transformation Matrix
S = make_S(listUnique, N);
eig_iterations = 100;  % Controls number of iterations for S'*S eigendecomposition approximation

%% Eigendecomposition
opts = struct;

eig_timer = tic;
fprintf('Begin Eigendecomposition of S^T*S ...\n');
opts.lambda = power_iteration(S'*S, eig_iterations);
eig_time_elapsed = toc(eig_timer);
fprintf('... End of Eigendecomposition. Time Elapsed: %.2f\n', eig_time_elapsed);

%% Learning Algorithm Call

opts.iter_max = 10000;  % Maximum algorithm iterations
opts.alpha = 0.1;  % Controls degree
opts.beta = 0.01;  % Controls sparsity
opts.eta = 1e-8;  % Controls algorithm stopping threshold
opts.epsilon_frac = 9/10;  % Multiplier for the learning rate range
opts.threshold = 1e-2;  % Threshold for learned weights w

[w, learned_edges, learned_weights, w_original]...
    = HSLS_algorithm_no_metrics(z, S, Da, N, listUnique, opts);

%% Check for Missing Nodes
% [learned_edges_updated,w_updated,learned_weights_updated,nodes_missing_initial] = missing_nodes(learned_edges,learned_weights,listUnique,w_original,w,N);

%% Compare Smoothness
learned_smoothness = sum(z(w > 0));  % Treat all weights as binary, so total variation is just sum of z

smoking_data = Smoking_Rates_Data_Processing();
temperature_data = Temperature_Data_Processing();

smoking_data_norm = zscore(smoking_data,1,'all');
temperature_data_norm = zscore(temperature_data,1,'all');

avg_data_smoking = mean(smoking_data_norm,2);
avg_data_temperature = mean(temperature_data_norm,2);

switch smooth_type
    case 1
        z_knn = smooth_square_sum(mortality_data_norm,C_cancer);  % Sum-Square distance vector
        z_smoking = smooth_square_sum(smoking_data_norm,learned_edges);
        z_temperature = smooth_square_sum(temperature_data_norm,learned_edges);
    case 2
        z_knn = smooth_abs_sum(mortality_data_norm,C_cancer);  % Sum-Absolute distance vector
        z_smoking = smooth_abs_sum(smoking_data_norm,learned_edges);
        z_temperature = smooth_abs_sum(temperature_data_norm,learned_edges);
    case 3
        z_knn = smooth_abs_max(mortality_data_norm,C_cancer);  % Max-Absolute distance vector
        z_smoking = smooth_abs_max(smoking_data_norm,learned_edges);
        z_temperature = smooth_abs_max(temperature_data_norm,learned_edges);
    case 4
        z_knn = smooth_square_max(mortality_data_norm,C_cancer);  % Max-Square distance vector
        z_smoking = smooth_square_max(smoking_data_norm,learned_edges);
        z_temperature = smooth_square_max(temperature_data_norm,learned_edges);
end

knn_smoothness = sum(z_knn);
smoking_smoothness = sum(z_smoking);
temperature_smoothness = sum(z_temperature);

fprintf('KNN Total Variation: %.3f\n', knn_smoothness)
fprintf('Learned Total Variation: %.3f\n', learned_smoothness)
fprintf('Smoking Total Variation: %.3f\n', smoking_smoothness)
fprintf('Temperature Total Variation: %.3f\n', temperature_smoothness)

%%
total_time = toc(global_timer);
fprintf('Total Time Elapsed: %.2f\n', total_time);