function [max_err,max_idx,rel_ls_err] = sd_paaa_errors(bf,samples,sampling_values,sparse_masking_idc,grid_size,nodes_part,itpl_subs,norm_2_samples,options)
%SD_PAAA_ERRORS Compute approximation error for sd_paaa greedy selection.

num_vars = size(sampling_values,2);

% error for greedy selection
err = abs(samples-bf.eval(sampling_values));
sparse_err = sparse(sparse_masking_idc,1,err,prod(grid_size),1);

% set errors to zero to enforce max_nodes constraint
if any(cellfun(@length,nodes_part) >= options.max_nodes)
    zero_idx = cell(1,num_vars);
    for i = 1:num_vars
        if length(nodes_part{i}) >= options.max_nodes(i)
            zero_idx{i} = 1:grid_size(i);
            if strcmp(options.itpl_type,'single')
                zero_idx{i}(nodes_part{i}) = [];
            end
        else
            zero_idx{i} = nodes_part{i};
        end
    end
    % determine indices for zero errors within vectorized error tensor
    [zero_grids{1:num_vars}] = ndgrid(zero_idx{:});
    zero_grids = cellfun(@(x) x(:), zero_grids, 'UniformOutput', false);
    zero_idc = sub2ind(grid_size, zero_grids{:});
    sparse_err(zero_idc) = 0;
elseif strcmp(options.itpl_type,'none')
    [zero_grids{1:num_vars}] = ndgrid(nodes_part{:});
    zero_grids = cellfun(@(x) x(:), zero_grids, 'UniformOutput', false);
    zero_idc = sub2ind(grid_size, zero_grids{:});
    sparse_err(zero_idc) = 0;
end

% set errors of interpolated data to zero
sparse_err(itpl_subs) = 0;
greedy_err = sparse_err(sparse_masking_idc);

% relative LS error
rel_ls_err = norm(err(:))^2 / norm_2_samples;

% maximum error for information
[max_err,~] = max(err,[],'all');

% maximum error for greedy
[~,max_idx] = max(greedy_err,[],'all');

end