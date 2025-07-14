function C = cauchy_mat(s,sigma,rem_sing_tol)
%CAUCHY_MAT Compute Cauchy-type matrix.

if nargin < 3
    rem_sing_tol = 1e-14;
end

D = s - sigma.';
[DZI1,DZI2] = find(abs(D) < rem_sing_tol);
C = 1 ./ D;

if numel(DZI1) > 0
    for j = 1:length(DZI1)
        C(:,DZI2(j)) = double(1:size(C,1) == DZI1(j));
    end
end

end
