function [w, learned_edges, learned_weights, w_original]...
    = HSLS_algorithm_no_metrics(z, S, Da, N, listUnique, opts)
%% Performs the FBF algorithm for the static hypergraph learning model
%
% Inputs:
%   z - Pairwise distance vector. Dimensions Da x 1
%   S - Matrix that satisfies S*w = d turning w into degree vector d. Matrix of size N x Da
%   Da - Positive integer number of unique hyperedge combinations
%   N - Positive integer number of nodes
%   listUnique - Cell array of vectors of all possible hyperedge combinations. Size Da x 1
%   opts - Structure of parameters to be set. This includes:
%               opts.iter_max: Positive integer upper bound for algorithmiterations
%               opts.alpha: Positive parameter for degree positivity term (negative log)
%               opts.beta: Positive parameter for sparsity term (L2 norm)
%               opts.eta: Algorithm stopping condition. The calculation must be less than this eta to break
%               opts.epsilon_frac: Value in range (0.0,1.0] that controls the starting point of the learning rate sequence. A smaller
%                                  value starts at a lower learning rate over a larger sequence range. A larger value starts at a higher
%                                  learning rate over a smaller sequence range.
%               opts.threshold: Positive value where anything smaller in the learned vector is set to 0.
%               opts.lambda: Eigenvalue from S'*S used for learning rate sequence
%
% Outputs:
%   w - Learned weight vector after being thresholded. Size Da x 1 
%   learned_edges - Cell array of vectors that are the hyperedges with nonzero weights from w. Size is number of learned edges x 1
%   learned_weights - The weights from w that correspond to learned_edges. Size is number of learned edges x 1
%   w_original - Learned weight vector prior to being thresholded. Size Da x 1

%% Instantiate  Constants
iter_max = opts.iter_max;
alpha = opts.alpha;
beta = opts.beta;
eta = opts.eta;

w = zeros(Da,1);  % Primal variable
d = zeros(N,1);  % Dual variable

%% Make Learning Rate Sequence
if isfield(opts, 'lambda')
    lambda_approx = opts.lambda;
    lip = 2*beta;
    mu = lip + sqrt(lambda_approx);
    epsilon = (opts.epsilon_frac) * (1/(1+mu));

    steps = (((1-epsilon) / mu) - epsilon) / (iter_max-1);
    gamma = epsilon:steps:(1-epsilon) / mu;
else
    gamma = opts.learning_rate * ones(1,iter_max);
end

%% FBF Algorithm Iterations
fbf_timer = tic;
fprintf('Begin FBF algorithm...\n');
for i = 1:iter_max
    if mod(i,100) == 0
        fprintf('Iteration %i\n', i)
    end

    y_1n = w - gamma(i)*(2*beta*w + S'*d);
    y_2n = d + gamma(i)*(S*w);
    p_1n = max(0,y_1n - gamma(i)*z);
    p_2n = y_2n - gamma(i)*(((y_2n./gamma(i)) + sqrt((y_2n./gamma(i)).^2 + 4*alpha/gamma(i))) ./ 2);
    q_1n = p_1n - gamma(i)*(2*beta*p_1n + S'*p_2n);
    q_2n = p_2n + gamma(i)*(S*p_1n);
    w_prev = w;
    d_prev = d;
    w = w - y_1n + q_1n;
    d = d - y_2n +q_2n;

    if ((w - w_prev)'*(w-w_prev) / (w_prev'*w_prev) < eta) && ((d - d_prev)'*(d-d_prev) / (d_prev'*d_prev) < eta)
        fprintf('Break reached on iteration %i\n', i)
        break
    end
end
fbf_time_elapsed = toc(fbf_timer);
fprintf('...End of algorithm. Time Elapsed: %.2f\n', fbf_time_elapsed);

%% Results

threshold = opts.threshold;
w_original = w;
w(w < threshold) = 0;

learned_edges = listUnique(w > 0);  % Cell array of learned hyperedges
learned_weights = w(w > 0);  % Vector of learned weights, corresponds to the learned_edges

end