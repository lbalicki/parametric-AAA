function [A,B,C,E] = bf_realization(alpha,beta,nodes)
% BF_REALIZATION Construct system realization of a barycentric form.
%
%   [A, B, C, E] = BF_REALIZATION(ALPHA, BETA, NODES)
%   Computes matrices A, B, C and E such that: r(z) = C' * (z * E - A)^(-1) * B.
%       
%   Inputs:
%       ALPHA    - Vector of barycentric denominator coefficients.
%       BETA     - Vector of barycentric numerator coefficients.
%       NODES    - Vector of barycentric nodes.
%
%   Outputs:
%       A        - System matrix A.
%       B        - System matrix B.
%       C        - System matrix C.
%       E        - System matrix E.

n = length(alpha);

E = zeros(n);
E(1:end-1,1) = ones(n-1,1);
E(1:end-1,2:end) = -eye(n-1);

A = zeros(n);
A(1:end-1,1) = nodes(1);
A(1:end-1,2:end) = -diag(nodes(2:end));
A(end,:) = alpha;

B = zeros(n,1);
B(end) = -1;

C = conj(beta);

end

