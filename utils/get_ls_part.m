function ls_part = get_ls_part(itpl_part, sampling_values)
% GET_LS_PART Based on a given partition of interpolation values compute the complement corresponding to the LS data partition.

ls_part = cell(1,length(itpl_part));
for i = 1:length(sampling_values)
    lp = 1:length(sampling_values{i});
    lp(itpl_part{i}) = [];
    ls_part{i} = lp;
end

end

