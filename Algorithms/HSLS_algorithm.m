function [w, learned_edges, learned_weights, overall_stats, w_original, w_from_t, C_organized] = HSLS_algorithm(z, S, Da, N, H, listUnique, opts)
% HSLS_algorithm (Run Li DHSLS-style, stacked incidence)
% FBF primal-dual on directed hypergraph with stacked incidence S = [S_in; S_out].
% Accepts S as (2N x Da) OR legacy (N x Da); legacy will be upgraded in-place.

    % -------------------- normalize z --------------------
    z = double(z(:));
    if numel(z) ~= Da, error('z must be Da x 1.'); end

    % -------------------- ensure S is stacked --------------------
    [Sr, Sc] = size(S);
    if Sc ~= Da, error('size(S,2) ~= Da'); end
    if Sr == N
        % upgrade legacy S (N x Da) -> (2N x Da) using listUnique [tails..., head]
        Sin  = sparse(N, Da);  Sout = sparse(N, Da);
        for e = 1:Da
            ed = listUnique{e}(:).';
            if numel(ed) < 2, continue; end
            head = ed(end);  tails = ed(1:end-1);
            tails = tails(tails ~= head & tails>=1 & tails<=N);
            if head>=1 && head<=N, Sin(head,e)=1; end
            if ~isempty(tails),     Sout(tails,e)=1; end
        end
        S_stacked = [Sin; Sout];
        Sr = size(S_stacked,1);
    elseif Sr == 2*N
        S_stacked = S;
    else
        error('S must be N x Da or (2N) x Da. Got %d x %d.', Sr, Sc);
    end
    N_from_S = Sr/2;
    if N_from_S ~= N, N = N_from_S; end

    % -------------------- options --------------------
    iter_max = opts.iter_max; alpha = opts.alpha; beta = opts.beta; eta = opts.eta;

    % -------------------- variables --------------------
    w = zeros(Da,1);
    d = zeros(2*N,1);

    % -------------------- stepsize schedule --------------------
    if isfield(opts,'lambda')
        lambda_approx = opts.lambda;
        lip = 2*beta; mu = lip + sqrt(lambda_approx);
        eps0 = (opts.epsilon_frac) * (1/(1+mu));
        if iter_max<=1
            gamma_seq = eps0;
        else
            steps = (((1-eps0)/mu) - eps0) / (iter_max-1);
            gamma_seq = (eps0:steps:(1-eps0)/mu).';
        end
    else
        if ~isfield(opts,'learning_rate'), error('Need opts.lambda or opts.learning_rate'); end
        gamma_seq = repmat(opts.learning_rate, max(1,iter_max), 1);
    end

    % -------------------- FBF iterations --------------------
    fbf_timer = tic; fprintf('Begin FBF algorithm (stacked S, %d x %d)...\n', size(S_stacked,1), size(S_stacked,2));
    for i = 1:iter_max
        if mod(i,100)==0, fprintf('Iteration %d\n', i); end
        gi = gamma_seq(min(i,numel(gamma_seq)));

        % forward
        y_1n = w - gi*( 2*beta*w + S_stacked.'*d );
        y_2n = d + gi*( S_stacked*w );

        % prox of f
        p_1n = max(0, y_1n - gi*z);

        % prox of g*
        t = y_2n./gi;
        p_2n = y_2n - gi * ( (t + sqrt(t.^2 + 4*alpha/gi)) / 2 );

        % backward
        q_1n = p_1n - gi*( 2*beta*p_1n + S_stacked.'*p_2n );
        q_2n = p_2n + gi*( S_stacked*p_1n );

        % update
        w_prev = w; d_prev = d;
        w = w - y_1n + q_1n;
        d = d - y_2n + q_2n;

        % stop
        if ((w-w_prev)'*(w-w_prev) / max(realmin, w_prev'*w_prev) < eta) && ...
           ((d-d_prev)'*(d-d_prev) / max(realmin, d_prev'*d_prev) < eta)
            fprintf('Break reached on iteration %d\n', i); break;
        end
    end
    fprintf('...End of algorithm. Time Elapsed: %.2f sec\n', toc(fbf_timer));
    iter_reached = i;

    % -------------------- post-processing --------------------
    threshold  = opts.threshold;
    w_original = w;
    w(w < threshold) = 0;

    % ----- TRY old metrics; FALLBACK to minimal outputs if it fails -----
    try
        C = cells_from_incidence(H);  % 旧评估链（可能不兼容你的 H 或 listUnique）
        [w_from_t, C_organized, learned_edges, learned_weights, A, precision, recall, F1, ~] = ...
            ground_truth_metrics(C, listUnique, w);
    catch ME
        warning('ground_truth_metrics failed: %s\nUsing minimal outputs (no GT metrics).', ME.message);
        % 只返回学习到的边与权重；GT 相关设为 NaN/空
        idx = find(w > 0);
        learned_edges   = listUnique(idx);
        learned_weights = w(idx);
        C_organized = {};
        w_from_t    = [];
        A = NaN; precision = NaN; recall = NaN; F1 = NaN;
    end

    overall_stats = [A; precision; recall; F1; alpha; beta; iter_reached];
end
