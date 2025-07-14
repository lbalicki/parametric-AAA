function [JD,sampling_values,samples] = init_JD(sampling_values,samples)
%INIT_JD Compute left realness enforcing unitary transform and initialize complex conjugate data.

ccp = cplxpair(sampling_values{1});
% [~,ccp_perm] = ismember(ccp,sampling_values{1});
[~,ccp_perm] = ismembertol([real(ccp);imag(ccp)].',[real(sampling_values{1});imag(sampling_values{1})].',100*eps(1),'ByRows',true);
sampling_values{1} = sampling_values{1}(ccp_perm);
samples(:,:) = samples(ccp_perm,:);

J = (1/sqrt(2)) * [1 1;-1i 1i];

% construct left projection matrix to make real Loewner matrix
first_real_idx = find(abs(imag(sampling_values{1})) < 100*eps(1), 1);
if isempty(first_real_idx)
    first_real_idx = length(sampling_values{1})+1;
end
JJ = kron(eye((first_real_idx-1)/2),J);
JD = blkdiag(JJ,eye(length(sampling_values{1})-first_real_idx+1));
end

