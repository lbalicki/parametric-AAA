function [A,E] = bf_compan(alpha,nodes)
% BF_REALIZATION Construct companion-type matrices of a barycentric form.
%
%   [A, E] = BF_REALIZATION(ALPHA, NODES)
%   Computes matrices A, E such that poles are given by eigenvalues e solving A x = e E x.
%
%   Inputs:
%       ALPHA    - Vector of barycentric denominator coefficients.
%       NODES    - Vector of barycentric nodes.
%
%   Outputs:
%       A        - System matrix A.
%       E        - System matrix E.

n = size(alpha,1);

E = zeros(n);
E(1:end-1,1) = ones(n-1,1);
E(1:end-1,2:end) = -eye(n-1);

A = zeros(n);
A(1:end-1,1) = nodes(1);
A(1:end-1,2:end) = -diag(nodes(2:end));
A(end,:) = alpha;

end

