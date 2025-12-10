function learned_edges = Gao_KNN_algorithm(X_v,K)
%% 

listUnique = [];

for i = 1:length(K)
    m = K(i);  % Current cardinality
    idx_raw = knnsearch(X_v, X_v,'K',m);
    idx = unique(sort(idx_raw,2),'rows');
    C = num2cell(idx,2);
    listUnique = [listUnique;C];
end

learned_edges = listUnique;

end