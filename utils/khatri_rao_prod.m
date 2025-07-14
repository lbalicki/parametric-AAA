function X = khatri_rao_prod(A,B)
%KHATRI_RAO_PROD Khatri-Rao product (column-wise Kronecker product) of two matrices.

[m, ~] = size(A);
[n, r] = size(B);

X = reshape(B, n, 1, r) .* reshape(A, 1, m, r);
X = reshape(X, m * n, r);
end

