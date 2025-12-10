clear
global_timer = tic;

%% Generate Signal
voltage = readmatrix('nodeVoltAllIEEE34cktnew.txt');

N = size(voltage,1); % Number of nodes
M = 3;  % Max cardinality

avg = mean(voltage(:));
std = std(voltage(:));
X_v = (voltage - avg) ./ std;  % Normalized via z-score

%% Unique list of hyperedge possibilities
all_hyperedge_possibilities = false;  % Selecting whether to use every hyperedge possibility or a K-NN shortened list
neighbors = [0,0,10,0,0];  % Selects number of neighbors per cardinality if all_hyperedge_possibilities is false
if all_hyperedge_possibilities
    count = 1:N;
    Da = nchoosek(N,M);  % Number of potential hyperedges
    edge_cell = num2cell(nchoosek(count,M),2);
    listUnique = edge_cell;  % Cell array of all hyperedge permutations
else
    neighbors_time = tic;
    listUnique = generate_knn_hyperedges(X_v,neighbors);
    Da = size(listUnique,1);
    neighbors_time_elapsed = toc(neighbors_time);
    fprintf('Nearest neighbor hyperedges time elapsed: %.2f\n', neighbors_time_elapsed)
end

%% Distance Vector
smooth_type = 4;
switch smooth_type
    case 1
        z = smooth_square_sum(X_v,listUnique);  % Distance vector squared sum
    case 2
        z = smooth_abs_sum(X_v,listUnique);  % Distance vector absolute sum
    case 3
        z = smooth_abs_max(X_v,listUnique);  % Distance vector absolute maximum
    case 4
        z = smooth_square_max(X_v,listUnique);  % Distance vector squared maximum
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
opts.alpha = 10;  % Controls degree
opts.beta = 0.01;  % Controls sparsity
opts.eta = 1e-8;  % Controls algorithm stopping threshold
opts.epsilon_frac = 9/10;  % Multiplier for the learning rate range
opts.threshold = 1e-2;  % Threshold for learned weights w

[w, learned_edges, learned_weights, w_original]...
    = HSLS_algorithm_no_metrics(z, S, Da, N, listUnique, opts);

%%
total_time = toc(global_timer);
fprintf('Total Time Elapsed: %.2f\n', total_time);