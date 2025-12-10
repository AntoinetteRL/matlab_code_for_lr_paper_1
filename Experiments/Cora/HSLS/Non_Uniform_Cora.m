% Non_Uniformed_Cora (Directed DHSLS-style main script)
clear
global_timer = tic;

%% 1) Load Ground Truth from a given path (incidence matrix H)
% Expecting a .mat file that contains a 99x99 incidence matrix variable.
% We will pick the first matrix-like variable found and use it as H.
mat_path = 'C:\Users\dell\Desktop\vector.mat';
S_loaded = load(mat_path);

% Pick the first numeric/logical 2-D variable as H
fn = fieldnames(S_loaded);
H = [];
for k = 1:numel(fn)
    cand = S_loaded.(fn{k});
    if isnumeric(cand) || islogical(cand)
        if ismatrix(cand)
            H = cand;
            break;
        end
    end
end
if isempty(H)
    error('No 2-D numeric/logical matrix found in %s to serve as incidence H.', mat_path);
end

% Ensure H is logical incidence (0/1)
if ~islogical(H)
    H = H ~= 0;
end

N = size(H, 1);                 % nodes
M = 3;                          % max cardinality (non-uniform uses {2,3})

%% 2) Generate Signals from H
observations = 250;             % number of signal observations
L = incidence_laplacian(H);     % Laplacian from incidence
[X_v, ~] = Bipartite_Signal(L, observations, N);

%% 3) Candidate Directed Hyperedge Set (KNN reduction)
all_hyperedge_possibilities = false;
neighbors = [5, 6];             % K for size=2 and size=3
if all_hyperedge_possibilities
    % Full (non-uniform: sizes 2..M). NOTE: these are undirected sets.
    % If you truly need full *directed* enumeration, expand per your rule.
    listUnique = [];
    count = 1:N;
    Da = 0;
    for i = 2:M
        Da = Da + nchoosek(N, i);
        edge_cell = num2cell(nchoosek(count, i), 2);
        listUnique = [listUnique; edge_cell]; %#ok<AGROW>
    end
else
    % Our directed KNN candidates: size=2 -> [tail head]; size=3 -> [tail1 tail2 head]
    listUnique = generate_knn_hyperedges(X_v, neighbors);
    Da = size(listUnique, 1);
end

%% 4) Directed TV vector z
% smooth_type: 1 = square-sum, 2 = abs-sum, 3 = abs-max, 4 = square-max
smooth_type = 4;
switch smooth_type
    case 1
        z = smooth_square_sum_directed(X_v, listUnique);
    case 2
        z = smooth_abs_sum_directed(X_v, listUnique);
    case 3
        z = smooth_abs_max_directed(X_v, listUnique);
    case 4
        z = smooth_square_max_directed(X_v, listUnique);
end

z = z(:);
if length(z) ~= Da
    error('Length of z (%d) does not match Da (%d).', length(z), Da);
end

%% 5) Stacked incidence S = [S_in; S_out]  (2N x Da)
S = make_S(listUnique, N);      % our make_S returns stacked S


% --- 兼容旧版 make_S（N x Da），在此就地升级成 (2N) x Da ---
if size(S,1) == N && size(S,2) == Da
    Da_local = size(S,2);
    Sin  = sparse(N, Da_local);
    Sout = sparse(N, Da_local);
    for e = 1:Da_local
        edge = listUnique{e}(:).';
        if numel(edge) < 2, continue; end
        head  = edge(end);
        tails = edge(1:end-1);
        tails = tails(tails ~= head);
        if head >= 1 && head <= N
            Sin(head, e) = 1;
        end
        tails = tails(tails >= 1 & tails <= N);
        if ~isempty(tails)
            Sout(tails, e) = 1;
        end
    end
    S = [Sin; Sout];  % 现在是 (2N) x Da
end

% 仍做安全检查
if size(S,1) ~= 2*N || size(S,2) ~= Da
    error('S must be (2N) x Da. Got %d x %d.', size(S,1), size(S,2));
end


%% 6) Eigendecomposition for stepsize schedule
eig_iterations = 100;
opts = struct;
eig_timer = tic;
fprintf('Begin Eigendecomposition of S^T*S ...\n');
opts.lambda = power_iteration(S'*S, eig_iterations);
eig_time_elapsed = toc(eig_timer);
fprintf('... End of Eigendecomposition. Time Elapsed: %.2f\n', eig_time_elapsed);

%% 7) Optimization Hyperparameters
opts.iter_max      = 10000;
opts.alpha         = 100;
opts.beta          = 0.1;
opts.eta           = 1e-8;
opts.epsilon_frac  = 9/10;
opts.threshold     = 1e-2;

%% 8) Run DHSLS-style HSLS_algorithm (stacked S)
[w, learned_edges, learned_weights, overall_stats, w_original, w_from_t, C_organized] = ...
    HSLS_algorithm(z, S, Da, N, H, listUnique, opts);

%% 9) Timing
total_time = toc(global_timer);
fprintf('Total Time Elapsed: %.2f\n', total_time);




%% 10) 打印权重最大的前 4 个超边（tails -> head）
% learned_edges: 每条超边的节点索引（cell）
% learned_weights: 对应的权重（向量）

if isempty(learned_weights) || isempty(learned_edges)
    warning('learned_edges 或 learned_weights 为空，无法打印 Top-4 超边。');
else
    % 按权重从大到小排序
    [w_sorted, idx_sorted] = sort(learned_weights(:), 'descend');
    top_k = min(4, numel(w_sorted));   % 只取前 4 条，若不足 4 条就全打印

    fprintf('\nTop-%d learned directed hyperedges (tails -> head):\n', top_k);
    for i = 1:top_k
        e     = learned_edges{idx_sorted(i)};  % 取对应的超边
        head  = e(end);                        % 最后一个节点当作 head
        tails = e(1:end-1);                    % 前面的当作 tails
        fprintf('%2d) [%s] -> %d   (w = %.6f)\n', ...
            i, num2str(tails), head, w_sorted(i));
    end
end
