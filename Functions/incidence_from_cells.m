function [H,C] = incidence_from_cells(C)
% Generate hypergraph incidence matrix H from column vector of cells C

    row_of_nodes = sort(unique(cell2mat(C')));

    N = length(row_of_nodes);
    H = zeros(N,length(C));
    new_labels = 1:N;

    for i = 1:length(C)
        for j = 1:length(C{i})
           C{i}(j) = new_labels(row_of_nodes == C{i}(j));
        end

        col = zeros(N,1);
        col(C{i}) = 1;
        H(:,i) = col;

    end

end