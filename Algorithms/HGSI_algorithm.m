function [learned_edges,learned_weights,w,listUnique,z] = HGSI_algorithm(X_v,K,num_hyperedges)
%% Static Hypergraph Learning Algorithm from "Hypergraph Structure Inference From Data Under Smoothness Prior"
%
% Inputs:
%   X_v - Nodes time series / feature matrix. Of size N x number of observations
%   uniform - Binary value indicating uniform cardinality (1) or non-uniform cardinality (0)
%   K - Vector of scalar values indicating which cardinalities are in the hypergraph. Ex: [2,5] indicates 
%       the hypergraph is made up of hyperedges of cardinality 2 and cardinality 5.
%
% Outputs:
%   w - Vector of learned weight probabilities. Entries correspond to listUnique
%   listUnique - Cell array where each entry contains a vector of nodes making up a potential hyperedge
%   z - Vector of hyperedge smoothness values. Entries correspond to listUnique
%   learned_edges - Cell array where each entry is a vector of nodes in a hyperedge. Contains the final
%                   list of learned hyperedges.
%   learned_weights - Vector of learned weights corresponding to learned_edges

%% Isolate the Nearest Neighbors as Hyperedges
listUnique = [];
for i = 1:length(K)
    m = K(i);  % Current cardinality
    idx_raw = knnsearch(X_v, X_v,'K',m);
    idx = unique(sort(idx_raw,2),'rows');
    C = num2cell(idx,2);
    listUnique = [listUnique;C];
end

%% Create the Distance Vector using Potential Hyperedge List
z = [];
for num_hyperedge = 1:size(listUnique,1)
    L2_norm = [];
    iter = 2;
    cur_nodes = listUnique{num_hyperedge};
    cardinality = length(cur_nodes);
    for i = 1:cardinality-1
        for j = iter:cardinality
            diff = X_v(cur_nodes(i),:) - X_v(cur_nodes(j),:);
            L2_norm(end+1,1) = diff*diff';
        end
        iter = iter + 1;
    end
    z(end+1,1) = max(L2_norm);  % Holds the smoothness based on max difference for each hyperedge
end

%% Use Closed Form Solution

w = 1 ./ (z + 1);

%% 
learned_edges = {};
learned_weights = [];

if num_hyperedges > length(w)
    num_hyperedges = length(w);
end

for i = 1:num_hyperedges
    [val,idx] = max(w);
    learned_edges(end+1,1) = listUnique(idx);
    learned_weights(end+1,1) = val;
    w(idx) = 0;
end

end