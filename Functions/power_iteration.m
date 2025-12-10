function [lambda, v] = power_iteration(A, num_iterations)
    v = rand(size(A, 1), 1);  % Random initial vector
    v = v / norm(v);           % Normalize the vector

    for i = 1:num_iterations
        v = A * v;             % Multiply by the matrix
        v = v / norm(v);      % Normalize the vector
    end

    % Rayleigh quotient for the eigenvalue approximation
    lambda = v' * A * v;
end