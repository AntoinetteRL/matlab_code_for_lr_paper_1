function C = cells_from_incidence(H)
% Generate a column cell array with hyperedges from incidence matrix H

indexes = 1:size(H,1);
C = {};

for i = 1:size(H,2)
    edge_vec = [];
    edge_vec = indexes(H(:,i) > 0);
    C(end+1) = {edge_vec};
end

C = C';

end