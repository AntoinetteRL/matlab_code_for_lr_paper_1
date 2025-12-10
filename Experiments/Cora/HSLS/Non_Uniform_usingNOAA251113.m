%% Non_Uniform_usingNOAA251113.m
% 从 NOAA 温度信号直接学习有向超图结构（DHSLS-style）
% 关键：强制让"行=48个节点"，候选与结果均做越界清洗；基于 learned_weights 打印 Top-98

clear; clc;
global_timer = tic;

%% 0) 路径
addpath(genpath('E:\experimentAndres251112'));

%% 1) 读取输入信号 X
data_path = 'E:\experimentAndres251112\noaa_statewide_tavg_1990_2025.mat';
S_loaded  = load(data_path);

% 取第一个二维数值矩阵
X_raw = [];
fn = fieldnames(S_loaded);
for k = 1:numel(fn)
    cand = S_loaded.(fn{k});
    if (isnumeric(cand) || islogical(cand)) && ismatrix(cand)
        X_raw = double(cand);
        break;
    end
end
if isempty(X_raw)
    error('在 %s 中没有找到二维数值矩阵作为信号 X。', data_path);
end

% —— 强制让"行=48个节点"；哪一维等于48，就把它放到行；否则报错
[m,n] = size(X_raw);
if m == 48
    X_v = X_raw;
elseif n == 48
    X_v = X_raw.';    % 把 48 放到行
else
    error('输入矩阵为 %dx%d，但没有维度等于 48；请先检查/整理文件。', m, n);
end
N = 48;
P = size(X_v, 2);
fprintf('After orientation: X_v is %dx%d (N=%d, P=%d)\n', size(X_v,1), size(X_v,2), N, P);

% 标准化（每个节点 z-score）
X_v = X_v - mean(X_v, 2);
stdv = std(X_v, 0, 2); stdv(stdv==0) = 1;
X_v = X_v ./ stdv;

%% 2) 不使用真值 H（可留空）
H = [];

%% 3) 候选有向超边（KNN 缩减 -> 清洗）
neighbors  = [min(15, N-1), min(10, N-1)];    % 2元/3元
listUnique = generate_knn_hyperedges(X_v, neighbors);   % 你的工程函数
Da0 = numel(listUnique);

% —— 清洗：剔除越界/长度<2/重复节点
lu_clean = cell(0,1);
for e = 1:Da0
    ed = listUnique{e}(:).';
    ed = unique(ed, 'stable');     % 去重复但保持顺序（尾...尾，最后是头）
    if numel(ed) < 2, continue; end
    if any(ed < 1 | ed > N), continue; end
    lu_clean{end+1,1} = ed;        %#ok<AGROW>
end
listUnique = lu_clean;
Da = numel(listUnique);
if Da == 0
    error('候选超边清洗后为空；请调大 neighbors 或检查数据。');
end
fprintf('KNN candidates: raw=%d, after_clean=%d\n', Da0, Da);

%% 4) 有向 TV 向量 z
smooth_type = 4;   % 1: square-sum, 2: abs-sum, 3: abs-max, 4: square-max
switch smooth_type
    case 1, z = smooth_square_sum_directed(X_v, listUnique);
    case 2, z = smooth_abs_sum_directed(X_v, listUnique);
    case 3, z = smooth_abs_max_directed(X_v, listUnique);
    case 4, z = smooth_square_max_directed(X_v, listUnique);
    otherwise, error('Unknown smooth_type.');
end
z = z(:);
if numel(z) ~= Da, error('Length of z(%d) != Da(%d).', numel(z), Da); end

%% 5) 叠堆入射矩阵 S = [S_in; S_out]  (2N x Da)
S = make_S(listUnique, N);
if size(S,1) == N && size(S,2) == Da
    Sin  = sparse(N, Da);
    Sout = sparse(N, Da);
    for e = 1:Da
        ed   = listUnique{e}(:).';
        head = ed(end);
        tails = ed(1:end-1);
        tails = tails(tails~=head);
        if head>=1 && head<=N,      Sin(head,e) = 1; end
        tails = tails(tails>=1 & tails<=N);
        if ~isempty(tails),         Sout(tails,e) = 1; end
    end
    S = [Sin; Sout];
end
if size(S,1) ~= 2*N || size(S,2) ~= Da
    error('S 必须是 (2N x Da)，当前为 %d x %d。', size(S,1), size(S,2));
end

%% 6) 步长最大特征值粗估
eig_iterations = 100;
fprintf('Begin Eigendecomposition of S^T S ...\n');
opts        = struct;
opts.lambda = power_iteration(S'*S, eig_iterations);
fprintf('... End. lambda ≈ %.4g\n', opts.lambda);

%% 7) 超参数（与你之前一致）
opts.iter_max      = 10000;
opts.alpha         = 100;
opts.beta          = 0.1;
opts.eta           = 1e-8;
opts.epsilon_frac  = 9/10;
opts.threshold     = 1e-2;

%% 8) 求解
[w, learned_edges, learned_weights, overall_stats, w_original, w_from_t, C_organized] = ...
    HSLS_algorithm(z, S, Da, N, H, listUnique, opts);

%% 9) 结果二次清洗（确保 learned_edges/weights 一一对应且均在 1..N）
L0 = numel(learned_edges);
le_edges = cell(0,1);
le_w     = [];
for i = 1:L0
    ed = learned_edges{i}(:).';
    if numel(ed) < 2, continue; end
    if any(ed < 1 | ed > N), continue; end
    ed = unique(ed,'stable');
    if numel(ed) < 2, continue; end
    le_edges{end+1,1} = ed;                 %#ok<AGROW>
    le_w(end+1,1)     = learned_weights(i); %#ok<AGROW>
end
learned_edges   = le_edges;
learned_weights = le_w(:);
L = numel(learned_edges);

% —— 自检（不会再用 cell2mat 横向拼，改为分别取每条的 min/max）
mins = cellfun(@(e) min(e), learned_edges);
maxs = cellfun(@(e) max(e), learned_edges);
fprintf('learned_edges kept=%d (from %d), nodes range=[%d, %d], N=%d\n', ...
        L, L0, min(mins), max(maxs), N);

%% 10) 保存
outdir = fullfile(pwd, 'learned_results_NOAA');
if ~exist(outdir,'dir')
    [ok,msg,msgid] = mkdir(outdir);
    if ~ok
        warning('mkdir 失败：%s (%s)。改用临时目录。', msg, msgid);
        outdir = fullfile(tempdir, 'learned_results_NOAA');
        if ~exist(outdir,'dir'), mkdir(outdir); end
    end
end
save(fullfile(outdir,'result_NOAA_transpose.mat'), ...
    'w','learned_edges','learned_weights','overall_stats', ...
    'w_original','w_from_t','C_organized', ...
    'neighbors','smooth_type','opts','listUnique','S','X_v','-v7.3');

%% 11) 打印 Top-98（严格以 learned_weights 与 learned_edges 对齐）
[w_sorted, idx] = sort(learned_weights(:), 'descend');
top_k  = min(172, numel(w_sorted));
idxTop = idx(1:top_k);

fprintf('\nTop-%d directed hyperedges (tails -> head), by learned_weights:\n', top_k);
for i = 1:top_k
    e     = learned_edges{idxTop(i)};
    head  = e(end);
    tails = e(1:end-1);
    fprintf('%3d) [%s] -> %d   (w = %.6f)\n', i, num2str(tails), head, w_sorted(i));
end

fprintf('\nTotal Time Elapsed: %.2f sec\n', toc(global_timer));
fprintf('Saved to %s\n', fullfile(outdir,'result_NOAA_transpose.mat'));
