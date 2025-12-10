function listUnique2 = best_K_test(X_v,neighbors,C_organized)
%% best K test
listUnique2 = generate_knn_hyperedges(X_v,neighbors);
% Check for missing hyperedges from C_organized
missing_hyperedges = {};
for i = 1:length(C_organized)
    if ~any(cellfun(@(x) isequal(x, C_organized{i}), listUnique2))
        missing_hyperedges{end+1} = C_organized{i};
    end
end
% Display missing hyperedges
if ~isempty(missing_hyperedges)
    fprintf('Missing hyperedges:\n');
    disp(missing_hyperedges);
    fprintf('Overlap rate:%.2f%%\n',(1-length(missing_hyperedges)/length(C_organized))*100);
end
end