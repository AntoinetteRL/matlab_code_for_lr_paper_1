function [H,C,avg_data] = KNN_Ground_Truth(data, M)
%% Using data, creates the incidence matrix and cells corresponding to the K-NN hypergraph
%
% Inputs:
%   data: A matrix where each row is a node and each column is a hypergraph signal. Of size N x M for N nodes
%   M: Scalar maximum cardinality of hyperedges. Is used to generate uniform hyperedges of cardinality M
%
% Outputs:
%   H - Incidence matrix of K-NN hypergraph. Of size N x number of unique hyperedges
%   C - Cell array where each cell contains a vector corresponding to nodes in a hyperedge. Of size number of hyperedges
%   avg_data - Returns the average of data along the columns, so the average value per node. Vector of size N

avg_data = mean(data,2);

[idx, ~] = knnsearch(data, data, 'K', M);
idx = unique(sort(idx,2),'rows');

C = cell(size(idx,1),1);

for i = 1:size(idx,1)
    C(i) = {sort(idx(i,:))};
end

[H,C] = incidence_from_cells(C);

end
