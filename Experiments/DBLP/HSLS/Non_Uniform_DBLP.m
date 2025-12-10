clear
global_timer = tic;

%% Load Ground Truth
load('incidence_matrix_connected_1_numEdges=39_N=58_M=3.mat') % Loads incidence matrix H for the coauthorship network

N = size(H,1); % Number of nodes
M = 3;  % Max cardinality

%% Generate Signal
observations = 250;  % Number of signal observations
L = incidence_laplacian(H);  % Creates Laplacian from incidence matrix
[X_v,~] = Bipartite_Signal(L,observations,N);  % Returns matrix of signal observations

%% Unique list of hyperedge possibilities
all_hyperedge_possibilities = false;  % Selecting whether to use every hyperedge possibility or a K-NN shortened list
neighbors = [5,6];  % Selects number of neighbors per cardinality if all_hyperedge_possibilities is false
if all_hyperedge_possibilities
    listUnique = [];
    count = 1:N;
    Da = 0;
    for i=2:M
        Da=Da+nchoosek(N,i);
        edge_cell = num2cell(nchoosek(count,i),2);
        listUnique = [listUnique;edge_cell];
    end
else
    listUnique = generate_knn_hyperedges(X_v,neighbors);
    Da = size(listUnique,1);
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

%% Perform Eigendecompostion
opts = struct;

eig_timer = tic;
fprintf('Begin Eigendecomposition of S^T*S ...\n');
opts.lambda = power_iteration(S'*S, eig_iterations);
eig_time_elapsed = toc(eig_timer);
fprintf('... End of Eigendecomposition. Time Elapsed: %.2f\n', eig_time_elapsed);

%% Learning Algorithm Call
opts.iter_max = 10000;  % Maximum algorithm iterations
opts.alpha = 100;  % Controls degree
opts.beta = 0.01;  % Controls sparsity
opts.eta = 1e-8;  % Controls algorithm stopping threshold
opts.epsilon_frac = 9/10;  % Multiplier for the learning rate range
opts.threshold = 1e-2;  % Threshold for learned weights w

[w, learned_edges, learned_weights, overall_stats, w_original, w_from_t, C_organized]...
    = HSLS_algorithm(z, S, Da, N, H, listUnique, opts);

%%
total_time = toc(global_timer);
fprintf('Total Time Elapsed: %.2f\n', total_time);