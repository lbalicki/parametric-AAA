function [sampling_values,samples] = pair_cc_data(sampling_values,samples)
%PAIR_CC_DATA Pair complex conjugate data.

ccp = cplxpair(sampling_values{1});
% [~,ccp_perm] = ismember(ccp,sampling_values{1});
[~,ccp_perm] = ismembertol([real(ccp);imag(ccp)].',[real(sampling_values{1});imag(sampling_values{1})].',100*eps(1),'ByRows',true);
sampling_values{1} = sampling_values{1}(ccp_perm);
samples(:,:) = samples(ccp_perm,:);

end

