function [bf,info] = sd_paaa(samples,sampling_values,tol,options)
% SD_PAAA Multivariate rational approximation with the p-AAA algorithm and scattered sampling points.
%
%   BF = SD_PAAA(SAMPLES, SAMPLING_VALUES, TOL, OPTIONS)
%   Computes a multivariate rational approximant in terms of a barycentric form.
%
%   Inputs:
%       SAMPLES          - Vector of size (n x 1) containing the samples to be approximated.
%       SAMPLING_VALUES  - Sampling points of size (n x d), where d is the number of variables.
%       TOL              - Convergence tolerance for the maximum error (default: 1e-3).
%       OPTIONS          - Struct containing options:
%                            * options.itpl_type      - String that is either 'all', 'single' or 'none' (default: 'all'). If 'all', all possible
%                                                       points are interpolated. If 'single' only isolated interpolation points are added. If 
%                                                       'none', no interpolation points are added.
%                            * options.max_nodes      - Maximum number of interpolation points in each variable (default: size(samples) - 1).
%                            * options.max_iter       - Maximum number of iterations for p-AAA (default: based on sampling_values).
%                            * options.more_info      - Include barycentric forms from each iteration in the INFO output. (default: false)
%
%   Outputs:
%       BF               - Rational approximant as a BarycentricForm instance.
%       INFO             - Cell array with information about the approximation at each iteration.
%

if nargin < 4
    options = struct;
end

num_samples = size(sampling_values,1);
num_vars = size(sampling_values,2);

% set interpolation type
if ~isfield(options,'itpl_type')
    options.itpl_type = 'all';
else
    assert(ismember(options.itpl_type, {'all', 'single', 'none'}), 'options.itpl_type must be ''all'', ''single'', or ''none''.');
end

% construct cell arrays for sampling grid
sampling_grid = cell(1,num_vars);
masking_idc = zeros(size(sampling_values));

for i = 1:num_samples
    for j = 1:num_vars
        sv_idx = find(sampling_grid{j} == sampling_values(i,j));
        if sv_idx
            masking_idc(i,j) = sv_idx;
        else
            sampling_grid{j} = [sampling_grid{j},sampling_values(i,j)];
            masking_idc(i,j) = length(sampling_grid{j});
        end
    end
end

grid_size = cellfun(@length,sampling_grid);
cell_masking_idc = num2cell(masking_idc, 1);
% indices of sampled data within the vectorized sparse sample tensor
sparse_masking_idc = sub2ind(grid_size, cell_masking_idc{:});

% vectorized sparse sample tensor
sparse_samples = sparse(sparse_masking_idc,1,samples,prod(grid_size),1);

if ~isfield(options,'max_iter')
    options.max_iter = prod(grid_size)-1;
end

% set maximum number of interpolation points in each variable
if ~isfield(options,'max_nodes')
    options.max_nodes = grid_size - 1;
    if num_vars == 1
        options.max_nodes = options.max_nodes(1);
    end
end

if ~isfield(options,'more_info')
    options.more_info = false;
end

if options.more_info
    info.itpl_inds = {};
    info.bf_iterates = {};
end

max_samples = max(abs(samples),[],'all');
norm_2_samples = norm(samples(:))^2;

% set initial partition of barycentric nodes
nodes_part = cell(1,num_vars);
itpl_subs = [];

err = abs(samples-mean(samples,'all'));
[max_err,max_idx] = max(err,[],'all');
rel_ls_err = norm(err(:))^2 / norm_2_samples;
fprintf('SD p-AAA Initial       | rel max err %.3e | rel LS err %.3e\n', max_err/max_samples, rel_ls_err);

% save information about errors
info.rel_max_errors = max_err/max_samples;
info.rel_ls_errors = rel_ls_err;

% do this such that p-AAA does at least one iteration
max_err = Inf;

j = 0;

while max_err > max_samples * tol && j < options.max_iter

    j = j + 1;

    max_Idx = masking_idc(max_idx,:);

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
            nodes_part{i} = unique([nodes_part{i};max_Idx(1,i)]);
        end
    end

    % get cell arrays for barycentric nodes
    nodes = cellfun(@(sv, lp) sv(lp), sampling_grid, nodes_part, 'UniformOutput', false);

    % determine indices of barycentric nodes within vectorized sample tensor
    [nodes_grids{1:num_vars}] = ndgrid(nodes_part{:});
    nodes_grids = cellfun(@(x) x(:), nodes_grids, 'UniformOutput', false);
    nodes_subs = sub2ind(grid_size, nodes_grids{:});

    % determine indices of interpolation points
    if strcmp(options.itpl_type, 'all')
        itpl_subs = intersect(nodes_subs, sparse_masking_idc);
    elseif strcmp(options.itpl_type, 'single')
        cell_max_Idx = num2cell(max_Idx);
        max_Idx_sub = sub2ind(grid_size,cell_max_Idx{:});
        if ismember(max_Idx_sub,nodes_subs)
            itpl_subs = [itpl_subs; max_Idx_sub];
        end
    end

    % solve LS problem
    [denom_coefs,num_coefs] = solve_sd_ls(sparse_samples,sampling_values,nodes,sparse_masking_idc,itpl_subs,nodes_subs);

    % assemble barycentric form
    bf = BarycentricForm(nodes,num_coefs,denom_coefs);

    % carefully compute errors for greedy selection
    [max_err,max_idx,rel_ls_err] = sd_paaa_errors(bf,samples,sampling_values,sparse_masking_idc,grid_size,nodes_part,itpl_subs,norm_2_samples,options);

    % add output information
    info.rel_max_errors(end+1) = max_err / max_samples;
    info.rel_ls_errors(end+1) = rel_ls_err;
    
    if options.more_info
        info.bf_iterates{end+1} = bf;
        info.itpl_inds{end+1} = find(ismember(sparse_masking_idc, itpl_subs));
    end

    fprintf('SD p-AAA Iteration %3d | rel max err %.3e | rel LS err %.3e | num nodes [%s]\n', ...
    j, max_err/max_samples, rel_ls_err, sprintf('%g ', cellfun(@length,nodes_part)));

end
end

