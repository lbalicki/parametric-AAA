function [cc_samples,cc_sampling_values] = cc_data(samples,sampling_values)
%CC_DATA Extend samples such that complex conjugate data with respect to the first variable is added if necessary.

s = size(samples);
cc_samples = zeros([2*s(1),s(2:end)]);
cc_samples(1:s(1),:) = samples(:,:);
cc_samples(s(1)+1:end,:) = conj(samples(:,:));

cc_sampling_values = sampling_values;
cc_sampling_values{1} = [cc_sampling_values{1},conj(cc_sampling_values{1})];

end

