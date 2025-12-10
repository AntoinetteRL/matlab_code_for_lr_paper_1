function [A, precision, recall, F1] = metrics(result,C)
%% Computes the number of correct and incorrect learned hyperedges

% Output of form [correct;incorrect]
% Inputs:
%   result - A cell array with learned edges where every entry is a vector corresponding to a learned hyperedge. Of size R x 1 for R learned hyperedges
%   C - A cell array of the ground truth hyperedges. Each entry is a vector containing the nodes connected in a hyperedge. Of size L x 1 for L hyperedges.
%       Note that each vector should be arranged in increasing numerical order -> [2 3 1] should be [1 2 3].
%
% Outputs:
%   A - Vector containing the count of correct hyperedge predictions as the first element and incorrect predictions as the second. Of size 2 x 1.
%   precision - Scalar precision metric
%   recall - Scalar recall metric
%   F1 - Scalar F1 metric

correct = 0;  % Number of correct learned edges
incorrect = 0;  % Number of incorrect learned edges

for j = 1:length(result)

    prev_cor = correct;

    for i = 1:length(C)
        if length(result{j}) == length(C{i})  % If these two edges have the same cardinality ...
            if result{j} == C{i}  % Only evaluates to true if every value in both arguments are the same
                correct = correct + 1;
                break;
            end
        end
    end

    if prev_cor == correct  % Only true if a match was never found
        incorrect = incorrect + 1;
    end

end

A = [correct;incorrect];

TP = correct;  % True positives
FP = incorrect;  % False positives
FN = length(C) - correct;  % False negatives

precision = TP / (TP + FP);
recall = TP / (TP + FN);
F1 = (2 * precision * recall) / (precision + recall);
end