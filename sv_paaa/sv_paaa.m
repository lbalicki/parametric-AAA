function [bf,info] = sv_paaa(samples,sampling_values,tol,options)
% SV_PAAA Multivariate rational approximation with the set-valued p-AAA algorithm.
%
%   BF = SV_PAAA(SAMPLES, SAMPLING_VALUES, TOL, OPTIONS)
%   Computes a multivariate rational approximant in terms of a barycentric form.
%
%   Inputs:
%       SAMPLES          - Multidimensional array of size N_1 x ... x N_d x n x m containing the samples of size n x m to be approximated.
%       SAMPLING_VALUES  - Cell array of sampling points in each variable such that size(sampling_values{i},2) == N_ for i = 1,...,d.
%       TOL              - Convergence tolerance for the maximum error (default: 1e-3).
%       OPTIONS          - Struct containing options:
%                            * options.nodes_part                   - Cell array of initial nodes (default: empty).
%                            * options.max_nodes                    - Maximum number of nodes in each variable (default: size(samples) - 1).
%                            * options.max_iter                     - Maximum number of iterations for p-AAA (default: based on sampling_values).
%                            * options.validation.samples           - Samples used to compute a validation error (default: N\A)
%                            * options.validation.sampling_values   - Sampling values used to compute a validation error (default: N\A)
%                            * options.more_info                    - Whether or not to include all barycentric forms in info (default: false).
%
%   Outputs:
%       BF               - Rational approximant as a BarycentricForm instance.
%       INFO             - Cell array with information about the approximation at each iteration.
%

if nargin < 4
    options = struct;
end

num_vars = length(sampling_values);

% convert samples to vector values
size_vals = [size(samples),1];
size_vals = size_vals(num_vars+1:num_vars+2);
vec_samples = reshape(samples,[size(samples,1:num_vars),prod(size_vals)]);

% set initial interpolation partition
if ~isfield(options,'nodes_part')
    nodes_part = cell(1,num_vars);
else
    nodes_part = options.nodes_part;
end

% set maximum number of interpolation points in each variable
if ~isfield(options,'max_nodes')
    options.max_nodes = size(samples,1:num_vars) - 1;
    if num_vars == 1
        options.max_nodes = options.max_nodes(1);
    end
end

if ~isfield(options,'max_iter')
    options.max_iter = prod(cellfun(@length,sampling_values))-1;
end

if ~isfield(options,'more_info')
    options.more_info = false;
end


info.bf_iterates = {};
info.rel_max_errors = [];
info.rel_ls_errors = [];
info.rel_linearized_ls_errors = [];
info.rel_validation_max_errors = [];
info.rel_validation_ls_errors = [];

vecnorm_samples = vecnorm(vec_samples,2,num_vars+1);
max_samples = max(vecnorm_samples,[],'all');
norm_2_samples = norm(vecnorm_samples(:))^2;

err_mat = vecnorm((vec_samples-mean(vec_samples,1:num_vars)),num_vars+1);
[max_err,max_idx] = max(err_mat,[],'all');
rel_ls_err = norm(err_mat(:))^2 / norm_2_samples;
fprintf('SV p-AAA Initial       | rel max err %.3e | rel LS err %.3e\n', max_err/max_samples, rel_ls_err);

% do this such that p-AAA does at least one iteration
max_err = Inf;

max_Idx = cell(1,num_vars);
j = 0;

while max_err > max_samples * tol && j < options.max_iter

    j = j + 1;

    [max_Idx{:}] = ind2sub(size(samples,1:num_vars),max_idx);

    % check if maximum order has been reached
    add_itpl = cellfun(@(ip,mi)length(ip)<mi,nodes_part,num2cell(options.max_nodes));
    if ~any(add_itpl)
        fprintf('Reached maximum number of interpolation points \n')
        break
    end

    % add interpolation points
    for i = 1:num_vars
        % make sure to keep at least one sample in LS partition
        if add_itpl(i)
            nodes_part{i} = unique([nodes_part{i},max_Idx{i}]);
        end
    end

    % solve LS problem
    L = vec_data_loewner_mat(vec_samples,sampling_values,nodes_part);
    [~,~,X] = svd(L,0);
    denom_coefs = X(:,end);
    
    % set zero coefficients to machine epsilon to avoid numerical issues
    denom_coefs(denom_coefs == 0) = eps;

    rel_lin_ls_err = norm(L * denom_coefs)^2 / norm_2_samples;

    itpl_samples = samples(nodes_part{:},:,:);
    if num_vars > 1
        itpl_samples = permute(itpl_samples,[num_vars:-1:1,num_vars+1,num_vars+2]);
    end
    itpl_samples = reshape(itpl_samples,[],size_vals(1),size_vals(2));
    nodes = cellfun(@(sv, lp) sv(lp), sampling_values, nodes_part, 'UniformOutput', false);
    num_coefs = denom_coefs .* itpl_samples;
    bf = BlockBarycentricForm(nodes,num_coefs,denom_coefs);

    % greedy selection
    err_mat = vecnorm(vec_samples-reshape(bf.eval(sampling_values),size(vec_samples)),2,num_vars+1);

    % set errors to zero to avoid no interpolation points to be added
    zero_idx = cell(1,num_vars);
    for i = 1:num_vars
        if length(nodes_part{i}) >= options.max_nodes(i)
            zero_idx{i} = 1:size(samples,i);
        else
            zero_idx{i} = nodes_part{i};
        end
    end
    err_mat_greedy = err_mat;
    err_mat_greedy(zero_idx{:}) = 0;

    % maximum error for information
    [max_err,~] = max(err_mat,[],'all');

    % maximum error for greedy
    [~,max_idx] = max(err_mat_greedy,[],'all');

    % LS error
    rel_ls_err = norm(err_mat(:))^2 / norm_2_samples;

    info.rel_max_errors(end+1) = max_err / max_samples;
    info.rel_ls_errors(end+1) = rel_ls_err;
    info.rel_linearized_ls_errors(end+1) = rel_lin_ls_err;
    if options.more_info
        info.bf_iterates{end+1} = bf;
    end

    if isfield(options,'validation')
        validation_err_mat = abs(options.validation.samples - bf.eval(options.validation.sampling_values));
        info.rel_validation_max_errors(end+1) = max(validation_err_mat,[],'all') / max(abs(options.validation.samples),[],'all');
        info.rel_validation_ls_errors(end+1) = norm(validation_err_mat(:))^2 / norm(options.validation.samples(:))^2;
    end

    fprintf('SV p-AAA Iteration %3d | rel max err %.3e | rel LS err %.3e | rel lin LS err %.3e | num nodes [%s]\n', ...
    j, max_err/max_samples, rel_ls_err, rel_lin_ls_err, sprintf('%g ', cellfun(@length,nodes_part)));

end
end

