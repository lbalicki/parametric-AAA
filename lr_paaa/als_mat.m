function L = als_mat(samples,sampling_values,itpl_part,coefs,free_idx,real_transforms)
% ALS_MAT Constructs a matrix for the ALS iteration in the low-rank p-AAA algorithm.
%
%   L = ALS_MAT(SAMPLES, SAMPLING_VALUES, ITPL_PART, COEFS, FREE_IDX, REAL_TRANSFORMS)
%   constructs the Loewner-type matrix used in lr_paaa iteration
%
%   Inputs:
%       SAMPLES          - Tensor toolbox tensor of sample data to be approximated.
%       SAMPLING_VALUES  - Cell array of sampling points in each variable.
%       ITPL_PART        - Cell array of interpolation partitions in each variable.
%       COEFS            - Cell array of coefficient matrices.
%       FREE_IDX         - Index of the free variable in the ALS iteration.
%       REAL_TRANSFORMS  - Struct containing transformations for enforcing real Loewner matrix:
%                            * real_transforms.UL - Left transformation matrix.
%                            * real_transforms.UR - Right transformation matrix.
%
%   Outputs:
%       L               - Contracted Loewner matrix used in lr_paaa ALS iteration.
%

orders = cellfun("length", itpl_part);
coef_rank = size(coefs{1},2);
num_vars = length(sampling_values);
num_free = orders(free_idx)*coef_rank;

% need to flip the sample tensor orders due to Matlabs Kronecker ordering
H = permute(samples,num_vars:-1:1);
H_I = samples(itpl_part{:});
H_I = permute(H_I,num_vars:-1:1);

K = contraction_mat(coefs,free_idx);
C_K = tensor(K,[flip(orders),num_free]); % flip orders due to Kronecker ordering

if isfield(real_transforms,'UL') && isfield(real_transforms,'UR')
    C_K = ttm(C_K,real_transforms.UR,num_vars);
end

HI_K = H_I(:) .* reshape(double(C_K),[],num_free);
C_HI_K = tensor(HI_K,[flip(orders),num_free]); % flip orders due to Kronecker ordering

for j = 1:num_vars
    C = cauchy_mat_itpl(sampling_values{j},itpl_part{j});
    C_K = ttm(C_K, C.', num_vars - j + 1);
    C_HI_K = ttm(C_HI_K, C.', num_vars - j + 1);
end

LR = reshape(double(C_HI_K),[],num_free);
C_K = reshape(double(C_K),[],num_free);
LL = H(:) .* C_K;

L = LL - LR;

if isfield(real_transforms,'UL') && isfield(real_transforms,'UR')
    L = tensor(L,[flip(size(samples)),orders(free_idx),coef_rank]);
    L = ttm(L,real_transforms.UL,num_vars);
    L = reshape(double(L),numel(samples),num_free);
    L = real(L);
end

end
