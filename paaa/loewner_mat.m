function L = loewner_mat(samples,sampling_values,nodes_part,real_transforms)
%LOEWNER_MAT Compute the higher-order Loewner matrix.
%
%   L = LOEWNER_MAT(SAMPLES, SAMPLING_VALUES, ITPL_PART, REAL_TRANSFORMS)
%   Constructs the higher-order Loewner matrix.
%
%   Inputs:
%       SAMPLES          - Multidimensional array of sample data.
%       SAMPLING_VALUES  - Cell array of sampling points in each variable.
%       NODES_PART       - Cell array of interpolation partitions in each variable.
%       REAL_TRANSFORMS  - Optional struct containing transformations for enforcing real Loewner matrix:
%                            * real_transforms.UL - Left transformation matrix.
%                            * real_transforms.UR - Right transformation matrix.
%
%   Outputs:
%       L               - Higher-order Loewner matrix.
%

if nargin < 4
    real_transforms = struct;
end

num_vars = length(sampling_values);

H_itpl = samples(nodes_part{:});
if num_vars > 1
    H_itpl = permute(H_itpl,num_vars:-1:1);
    samples = permute(samples,num_vars:-1:1);
end

% indices of zero rows
z_idc = false(length(sampling_values{1}),1);
z_idc(nodes_part{1}) = true;

% Kronecker product of Cauchy-like matrices
kron_C = cauchy_mat_itpl(sampling_values{1},nodes_part{1}).';
for i = 2:num_vars
    % update Kronecker products
    C = cauchy_mat_itpl(sampling_values{i},nodes_part{i}).';
    kron_C = kron(kron_C,C);

    % update zero rows
    z_idc_i = false(length(sampling_values{i}),1);
    z_idc_i(nodes_part{i}) = true;
    z_idc = z_idc_i & z_idc.';
    z_idc = z_idc(:);
end

nnz_idc = ~z_idc;

% perform transformation to real loewner matrix
if isfield(real_transforms,'UL') && isfield(real_transforms,'UR')
    L = samples .* kron_C - (H_itpl(:) .* kron_C.').';
    L = reshape(L,[size(samples),size(H_itpl)]);
    L = tensorprod(real_transforms.UL,L,2,num_vars);
    L = permute(L, [2:num_vars,1,num_vars+1:2*num_vars]);
    L = tensorprod(real_transforms.UR.',L,2,2*num_vars);
    L = permute(L, [2:2*num_vars,1]);
    L = reshape(L,numel(samples),numel(H_itpl));
    L = real(L);
    L(z_idc,:) = [];
else
    L = samples(nnz_idc) .* kron_C(nnz_idc,:) - (H_itpl(:) .* kron_C(nnz_idc,:).').';
end

end

