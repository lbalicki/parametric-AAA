function JH = get_JH(sampling_values,itpl_part)
%GET_JH Compute right realness enforcing unitary transform.

J = (1/sqrt(2)) * [1 1;-1i 1i];

first_real_idx = find(abs(imag(sampling_values{1}(itpl_part{1}))) < 100*eps(1), 1);
if isempty(first_real_idx)
    first_real_idx = length(itpl_part{1})+1;
end
JJ = kron(eye((first_real_idx-1)/2),J);
JH = blkdiag(JJ,eye(length(itpl_part{1})-first_real_idx+1))';
end

