function bf = fit_sd_interp(samples,sampling_values,nodes,itpl_inds)
%FIT_SD_INTERP Fit a rational function to data with interpolation constraints.

num_samples = size(sampling_values,1);
num_vars = size(sampling_values,2);

% construct cell arrays for sampling grid
sampling_grid = cell(1,num_vars);
nodes_part = cell(1,num_vars);
masking_idc = zeros(size(sampling_values));

for j = 1:num_vars    
    for i = 1:num_samples
        sv_idx = find(sampling_grid{j} == sampling_values(i,j));
        if sv_idx
            masking_idc(i,j) = sv_idx;
        else
            sampling_grid{j} = [sampling_grid{j},sampling_values(i,j)];
            masking_idc(i,j) = length(sampling_grid{j});
        end
    end
    [~,nodes_part_j] = ismember(sampling_grid{j}, nodes{j});
    nodes_part{j} = find(nodes_part_j ~= 0);
end

grid_size = cellfun(@length,sampling_grid);
cell_masking_idc = num2cell(masking_idc, 1);

% indices of sampled data within the vectorized sparse sample tensor
sparse_masking_idc = sub2ind(grid_size, cell_masking_idc{:});

% vectorized sparse sample tensor
sparse_samples = sparse(sparse_masking_idc,1,samples,prod(grid_size),1);

cell_itpl_masking_idc = num2cell(masking_idc(itpl_inds,:), 1);
itpl_subs = sub2ind(grid_size, cell_itpl_masking_idc{:});

% determine indices of barycentric nodes within vectorized sample tensor
[nodes_grids{1:num_vars}] = ndgrid(nodes_part{:});
nodes_grids = cellfun(@(x) x(:), nodes_grids, 'UniformOutput', false);
nodes_subs = sub2ind(grid_size, nodes_grids{:});

% solve the ls problem as in scattered data p-AAA
[denom_coefs,num_coefs] = solve_sd_ls(sparse_samples,sampling_values,nodes,sparse_masking_idc,itpl_subs,nodes_subs);

bf = BarycentricForm(nodes,num_coefs,denom_coefs);

end

