function L = loewner_mat(samples,sampling_values,itpl_part,real_transforms)
%LOEWNER_MAT Compute the higher-order Loewner matrix.
%
%   L = LOEWNER_MAT(SAMPLES, SAMPLING_VALUES, ITPL_PART, REAL_TRANSFORMS)
%   Constructs the higher-order Loewner matrix.
%
%   Inputs:
%       SAMPLES          - Multidimensional array of sample data.
%       SAMPLING_VALUES  - Cell array of sampling points in each variable.
%       ITPL_PART        - Cell array of interpolation partitions in each variable.
%       REAL_TRANSFORMS  - Struct containing transformations for enforcing real Loewner matrix:
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

H_itpl = samples(itpl_part{:});
if num_vars > 1
    H_itpl = permute(H_itpl,num_vars:-1:1);
    samples = permute(samples,num_vars:-1:1);
end

kron_C = 1;
for i = 1:num_vars
    C = cauchy_mat_itpl(sampling_values{i},itpl_part{i}).';
    kron_C = kron(kron_C,C);
end

L = samples(:) .* kron_C - (H_itpl(:) .* kron_C.').';

if isfield(real_transforms,'UL') && isfield(real_transforms,'UR')
    L = reshape(L,[size(samples),size(H_itpl)]);
    L = tensorprod(real_transforms.UL,L,2,num_vars);
    L = permute(L, [2:num_vars,1,num_vars+1:2*num_vars]);
    L = tensorprod(real_transforms.UR.',L,2,2*num_vars);
    L = permute(L, [2:2*num_vars,1]);
    L = reshape(L,numel(samples),numel(H_itpl));
    L = real(L);
end

end

