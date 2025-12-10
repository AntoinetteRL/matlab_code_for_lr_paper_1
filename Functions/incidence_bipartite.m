function all_pairs = incidence_bipartite(H)
%% Used to pass to Python script for hypergraph plotting based on incidence matrix H


all_pairs = {};

for i = 1:size(H,2)
    cur_edge = H(:,i);
    node_vals = find(cur_edge);

    for j = 1:length(node_vals)
        all_pairs(1,end+1) = {{int64(node_vals(j)), strcat('e',num2str(i))}};
    end

end


end