function L = vec_data_loewner_mat(samples,sampling_values,nodes_part)
%VEC_DATA_LOEWNER_MAT Compute the higher-order Loewner matrix with vector-valued data.
%
%   L = VEC_DATA_LOEWNER_MAT(SAMPLES, SAMPLING_VALUES, ITPL_PART, REAL_TRANSFORMS)
%   Constructs the higher-order Loewner matrix with vector-valued data.
%
%   Inputs:
%       SAMPLES          - Tensor toolbox tensor of sample data to be approximated.
%       SAMPLING_VALUES  - Cell array of sampling points in each variable.
%       ITPL_PART        - Cell array of interpolation partitions in each variable.
%
%   Outputs:
%       L               - Higher-order Loewner matrix.
%

num_vars = length(sampling_values);
num_vals = size(samples,num_vars+1);

H_itpl = samples(nodes_part{:},:);
if num_vars > 1
    H_itpl = permute(H_itpl,[num_vars:-1:1,num_vars+1]);
    H_itpl = reshape(H_itpl,[],size(H_itpl,num_vars+1));
    samples = permute(samples,[num_vars:-1:1,num_vars+1]);
    samples = reshape(samples,[],size(samples,num_vars+1));
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
nnz_num = nnz(nnz_idc);

L = zeros(num_vals*nnz_num,size(kron_C,2));
for j = 1:num_vals
    L((j-1)*nnz_num+1:j*nnz_num,:) = samples(nnz_idc,j) .* kron_C(nnz_idc,:) - (H_itpl(:,j) .* kron_C(nnz_idc,:).').';
end


end

