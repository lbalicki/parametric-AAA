function L = vec_data_loewner_mat(samples,sampling_values,itpl_part)
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

H_itpl = samples(itpl_part{:},:);
if num_vars > 1
    H_itpl = permute(H_itpl,[num_vars:-1:1,num_vars+1]);
    H_itpl = reshape(H_itpl,[],size(H_itpl,num_vars+1));
    samples = permute(samples,[num_vars:-1:1,num_vars+1]);
    samples = reshape(samples,[],size(samples,num_vars+1));
end

kron_C = 1;
for i = 1:num_vars
    C = cauchy_mat_itpl(sampling_values{i},itpl_part{i}).';
    kron_C = kron(kron_C,C);
end

L = zeros(num_vals*size(kron_C,1),size(kron_C,2));
for j = 1:num_vals
    L((j-1)*size(kron_C,1)+1:j*size(kron_C,1),:) = samples(:,j) .* kron_C - (H_itpl(:,j) .* kron_C.').';
end


end

