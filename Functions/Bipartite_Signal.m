function [X_v,X] = Bipartite_Signal(L,observations,N)
%% Generates a hypergraph signal from the bipartite representation of the hypergraph

% Inputs:
%   L - Laplacian matrix of the bipartite representation of the hypergraph.
%   observations - Number of hypergraph signals to generate for L.
%   N - Number of nodes
%
% Outputs:
%   X_v - The signals corresponding to the nodes of the hypergraph. Of size N x observations
%   X - The signals corresponding to both the nodes and hyperedges. Of size (N + |E|) x observations where |E| is the number of hyperedges.

mean = 0;  % Mean of distribution
variance = 1e-3;  % Noise term
tolerance = 1e-6;  % Filters out terms smaller than this from a pseudo inverse operation

ones_vec = ones(size(L,1),1);

mu = mean * ones_vec;  % Vector with mean value
Sigma_raw = pinv(L) + diag(variance * ones_vec);  % Covariance matrix
Sigma = (Sigma_raw + Sigma_raw') ./ 2;  % Makes the matrix perfectly symmetric (avoids precision issue)
Sigma(abs(Sigma) < tolerance) = 0;  % Threshold the covariance matrix

X = mvnrnd(mu,Sigma,observations)';  % Generate signals from distribution

X_v = X(1:N,:);  % Use only signal values for hypergraph nodes
end