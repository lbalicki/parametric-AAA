function K = contraction_mat(coefs,free_idx)
% CONTRACTION_MAT Matrix required for contracting Loewner matrix for low-rank p-AAA LS problem.

r = size(coefs{1},2);
orders = cellfun(@(c)(size(c,1)), coefs);
K = zeros(prod(orders),orders(free_idx)*r);

for i = 1:r
    k_i = 1;
    for j = 1:length(coefs)
        if j == free_idx
            k_i = kron(k_i,eye(orders(j)));
        else
            k_i = kron(k_i,coefs{j}(:,i));
        end
    end
    K(:,(i-1)*orders(free_idx)+1:i*orders(free_idx)) = k_i;
end

end

