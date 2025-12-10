function [w_from_t,C_organized,learned_edges,learned_weights,A,precision,recall,F1,vech_t] = ground_truth_metrics(C,listUnique,w)

Da = length(listUnique);

vech_t = zeros(Da,1);  % Binary ground truth w
for i = 1:length(C)
    for j = 1:Da
        if length(listUnique{j}) == length(C{i})
            if  listUnique{j} == C{i}
                index = j;
            end
        end
    end
    vech_t(index) = 1;
end

w_from_t = w(vech_t > 0);  % Returns the values in learned w that correspond to the correct hyperedges
C_organized = listUnique(vech_t > 0);  % Cell array of the ground truth hyperedges

learned_edges = listUnique(w > 0);  % Cell array of learned hyperedges
learned_weights = w(w > 0);  % Vector of learned weights, corresponds to the learned_edges

[A, precision, recall, F1] = metrics(learned_edges,C);

end