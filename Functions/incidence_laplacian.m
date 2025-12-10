function L = incidence_laplacian(H)
% Creates the Laplacian matrix corresponding to the bipartite graph representation of hypergraph H

    num_edges = size(H,2);  % number of hyperedges
    num_nodes = size(H,1);  % number of nodes

    ones_edges = ones(num_edges,1);  % ones vector length of number of hyperedges
    ones_nodes = ones(num_nodes,1);  % ones vector length of number of nodes
    
    upper_left = diag(H*ones_edges);  % block upper left of L
    lower_right = diag(H'*ones_nodes);  % block lower right of L

    left = cat(1,upper_left,-H');  % Left side of L
    right = cat(1,-H,lower_right);  % Right side of L

    L = cat(2,left,right);  % Concatenating left and right side

end