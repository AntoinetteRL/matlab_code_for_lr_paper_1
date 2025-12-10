function listUnique = generate_knn_hyperedges(X, num_neighbors)
% Generate directed candidate hyperedges using KNN (size-2 and size-3 only).
% Size-2: [tail head], tail has larger signal than head.
% Size-3: [tail1 tail2 head], two larger-signal nodes jointly point to the smallest.
%
% Inputs:
%   X            : N x P matrix (N nodes, P-length features/time series)
%   num_neighbors: vector, t-th element is K for hyperedges of size (t+1)
%
% Output:
%   listUnique   : cell array (column). Each cell is a row vector of node indices.

    [N, ~] = size(X);
    if nargin < 2 || isempty(num_neighbors)
        listUnique = {};
        return;
    end

    % Scalar score to determine direction (larger -> smaller)
    score = mean(X, 2);

    % Cosine distance for similarity; remove self
    D = pdist2(X, X, 'cosine');
    D(1:N+1:end) = inf;

    % Global KNN order (ascending distance)
    kmax = max(num_neighbors(:));
    [~, order] = sort(D, 2, 'ascend');
    if N > 1
        NN = order(:, 1:min(kmax, N-1));
    else
        NN = zeros(N,0);
    end

    % Fast de-dup
    seen = containers.Map('KeyType','char','ValueType','logical');
    listUnique = {};

    % Only sizes 2 and 3 are generated per your rule
    max_size = min(3, length(num_neighbors)+1);

    for edge_size = 2:max_size
        k = num_neighbors(edge_size - 1);
        if k <= 0, continue; end

        % Ensure we have at least k neighbors available per node
        k_use = min(k, size(NN,2));
        if k_use == 0, continue; end

        if edge_size == 2
            % Pair edges: for each i, connect to its first k_use neighbors
            for i = 1:N
                neigh = NN(i, 1:k_use);
                for jj = 1:numel(neigh)
                    j = neigh(jj);
                    if j == i, continue; end
                    if score(i) >= score(j)
                        e = [i, j];   % [tail head]
                    else
                        e = [j, i];
                    end
                    key = sprintf('%d-%d', e(1), e(2));
                    if ~isKey(seen, key)
                        seen(key) = true;
                        listUnique{end+1} = e; %#ok<AGROW>
                    end
                end
            end

        elseif edge_size == 3
            % Triple edges: for each i, pick 2 among its first k_use neighbors
            for i = 1:N
                neigh = NN(i, 1:k_use);
                if numel(neigh) < 2, continue; end
                pairs = nchoosek(neigh, 2);
                for jj = 1:size(pairs, 1)
                    a = pairs(jj, 1);
                    b = pairs(jj, 2);
                    triple = [i, a, b];

                    % Two larger scores -> tails; smallest -> head
                    [~, ord] = sort(score(triple), 'descend');
                    tails = sort(triple(ord(1:2))); % tails in ascending index
                    head  = triple(ord(3));
                    e = [tails, head];              % [tail1 tail2 head]

                    key = sprintf('%d-%d-%d', e(1), e(2), e(3));
                    if ~isKey(seen, key)
                        seen(key) = true;
                        listUnique{end+1} = e; %#ok<AGROW>
                    end
                end
            end
        end
    end

    % Column cell array to match typical downstream expectations
    listUnique = listUnique.';
end
