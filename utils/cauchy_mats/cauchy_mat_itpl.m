function C = cauchy_mat_itpl(sampling_values,itpl_part)
%CAUCHY_MAT_ITPL Compute Cauchy-type matrix based on given interpolation partition.

N = size(sampling_values,1);
k = length(itpl_part);
ls_part = get_ls_part({itpl_part},{sampling_values});
ls_part = ls_part{1};

C = zeros(k,N);
C(:,itpl_part) = eye(k);
C(:,ls_part) = 1 ./ (sampling_values(ls_part) - sampling_values(itpl_part).');

end

