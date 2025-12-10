function [learned_edges_updated,w_updated,learned_weights_updated,nodes_missing_initial] = missing_nodes(learned_edges,learned_weights,listUnique,w_original,w,N)

nodes_missing = (1:N)';
matrix_edges = cell2mat(learned_edges);
nodes_used = unique(sort(matrix_edges,2));
nodes_missing(nodes_used) = [];
nodes_missing_initial = nodes_missing;

while ~isempty(nodes_missing)
    
    matrix_all_edges = cell2mat(listUnique);

    cur_node = nodes_missing(1);  % Current missing node
    cur_weights = w_original(any(matrix_all_edges == cur_node,2));  % Weights of every hyperedge containing current node
    potential_edges = listUnique(any(matrix_all_edges == cur_node,2));  % Hyperedges corresponding to the weights
    [val,idx] = max(cur_weights);  % Largest weight of any hyperedge with the current node in it
    learned_edges(end+1,:) = potential_edges(idx);  % Include hyperedge in the learned hyperedges
    learned_weights(end+1) = val;
    nodes_missing(1) = [];  % Delete the missing node from the list

    for i = 1:length(listUnique)
        if isequal(listUnique{i},potential_edges{idx})
            w(i) = val;  % Include the weight of the new hyperedge
            break
        end
    end

end

learned_edges_updated = learned_edges;
learned_weights_updated = learned_weights;
w_updated = w;

end