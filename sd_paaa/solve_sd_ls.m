function [denom_coefs,num_coefs] = solve_sd_ls(sparse_samples,sampling_values,nodes,sparse_masking_idc,itpl_subs,nodes_subs)
%SOLVE_SD_LS Solve scattered data p-AAA least squares problem.

num_vars = size(sampling_values,2);
node_grid_size = cellfun(@length,nodes);
num_nodes = prod(node_grid_size);

% get vector of samples
samples = full(sparse_samples(sparse_masking_idc));

itpl_mask = spalloc(size(sparse_samples,1),size(sparse_samples,2),length(itpl_subs));
itpl_mask = logical(itpl_mask);
itpl_mask(itpl_subs) = true;

% indices of zero/non-zero rows of LS matrix
z_idc = itpl_mask(sparse_masking_idc);
nnz_idc = ~z_idc;

% restrict interpolation mask to nodes
itpl_mask = itpl_mask(nodes_subs);

itpl_samples = spalloc(size(sparse_samples,1),size(sparse_samples,2),length(itpl_subs));
itpl_samples(itpl_subs) = sparse_samples(itpl_subs);
itpl_samples = itpl_samples(nodes_subs);

coefs_itpl_subs = 1:num_nodes;
coefs_itpl_subs(~itpl_mask) = [];

coefs_ls_subs = 1:num_nodes;
coefs_ls_subs(itpl_mask) = [];
num_ls = length(coefs_ls_subs);

kr_C = cauchy_mat(sampling_values(:,1).',nodes{1});
for i = 2:num_vars
    C = cauchy_mat(sampling_values(:,i).',nodes{i});
    kr_C = khatri_rao_prod(C,kr_C);
end

kr_C = kr_C(:,nnz_idc).';
kr_C_W = kr_C(:,coefs_ls_subs);

M = [samples(nnz_idc) .* kr_C - (itpl_samples .* kr_C.').', -kr_C_W];

% solve the LS problem
[~,~,V] = svd(M,0);

coefs = V(:,end);
denom_coefs = coefs(1:end-num_ls);

% set interpolated zero coefficients to machine epsilon to avoid undefined values
zero_mask = denom_coefs(coefs_itpl_subs) == 0;
denom_coefs(coefs_itpl_subs(zero_mask)) = eps;

ls_num_coefs = coefs(end-num_ls+1:end);
itpl_num_coefs = denom_coefs(coefs_itpl_subs) .* itpl_samples(coefs_itpl_subs);

num_coefs = zeros(size(denom_coefs));
num_coefs(coefs_ls_subs) = ls_num_coefs;
num_coefs(coefs_itpl_subs) = itpl_num_coefs;

denom_coefs = reshape(denom_coefs, node_grid_size);
num_coefs = reshape(num_coefs, node_grid_size);

end

