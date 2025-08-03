function [bf,info] = paaa(samples,sampling_values,tol,options)
% PAAA Multivariate rational approximation with the p-AAA algorithm.
%
%   BF = PAAA(SAMPLES, SAMPLING_VALUES, TOL, OPTIONS)
%   Computes a multivariate rational approximant represented in barycentric form.
%
%   Inputs:
%       SAMPLES          - Multidimensional array of size N_1 x ... x N_d containing the samples to be approximated.
%       SAMPLING_VALUES  - Cell array of sampling points in each variable such that size(sampling_values{i},2) == N_i.
%       TOL              - Convergence tolerance for the relative maximum error (default: 1e-3).
%       OPTIONS          - Struct containing options:
%                            * options.nodes_part                   - Cell array of initial nodes (default: empty).
%                            * options.itpl_cc                      - Whether or not to interpolate complex conjugate values (default: false). 
%                            * options.real_loewner                 - Whether or not to make Loewner matrix real-valued, requires that sampling_values
%                                                                     are complex conjugates in the first and real in other variables (default: false).
%                            * options.max_nodes                    - Maximum number of nodes in each variable (default: size(samples) - 1).
%                            * options.min_nodes                    - Minimum number of interpolation points in each variable (default: 0).
%                            * options.max_iter                     - Maximum number of iterations for p-AAA (default: based on sampling_values).
%                            * options.validation.samples           - Samples used to compute a validation error (default: N\A)
%                            * options.validation.sampling_values   - Sampling values used to compute a validation error (default: N\A)
%                            * options.more_info                    - Whether or not to include all barycentric forms in info (default: false).
%
%   Outputs:
%       BF               - Rational approximant as a BarycentricForm instance.
%       INFO             - Cell array with information about the approximation at each iteration.
%

if nargin < 4
    options = struct;
end

num_vars = length(sampling_values);

% set initial interpolation partition
if ~isfield(options,'nodes_part')
    nodes_part = cell(1,num_vars);
    num_nodes = zeros(1,num_vars);
else
    nodes_part = options.nodes_part;
    num_nodes = cellfun(@length,nodes_part);
end

if ~isfield(options,'real_loewner')
    options.real_loewner = false;
end

% set maximum number of interpolation points in each variable
if ~isfield(options,'max_nodes')
    options.max_nodes = size(samples) - 1;
    if num_vars == 1
        options.max_nodes = options.max_nodes(1);
    end
    if options.real_loewner
        options.max_nodes(1) = options.max_nodes(1) - 1;
    end
end

% set minimum number of interpolation points in each variable
if ~isfield(options,'min_nodes')
    options.min_nodes = zeros(1,num_vars);
end

if ~isfield(options,'max_iter')
    options.max_iter = prod(cellfun(@length,sampling_values))-1;
end

if ~isfield(options,'more_info')
    options.more_info = false;
end

if ~isfield(options,'itpl_cc')
    options.itpl_cc = false;
end

% in order to compute a real loewner matrix we need orthogonal transformation matrices  
% and must ensure that data appears in complex conjugate pairs
real_transforms = struct;
if options.real_loewner
    options.itpl_cc = true;
    [real_transforms.UL,sampling_values,samples] = init_JD(sampling_values,samples);
end

if options.itpl_cc
    [sampling_values,samples] = pair_cc_data(sampling_values,samples);
end

info.bf_iterates = {};
info.rel_max_errors = [];
info.rel_ls_errors = [];
info.rel_linearized_ls_errors = [];
info.rel_validation_max_errors = [];
info.rel_validation_ls_errors = [];

max_samples = max(abs(samples),[],'all');
norm_2_samples = norm(samples(:))^2;

err_mat = abs(samples-mean(samples,'all'));
[max_err,max_idx] = max(err_mat,[],'all');
rel_ls_err = norm(err_mat(:))^2 / norm_2_samples;
fprintf('p-AAA Initial       | rel max err %.3e | rel LS err %.3e\n', max_err/max_samples, rel_ls_err);

% do this such that p-AAA does at least one iteration
max_err = Inf;

max_Idx = cell(1,num_vars);
j = 0;

while (max_err > max_samples * tol && j < options.max_iter) || any(num_nodes<options.min_nodes)

    j = j + 1;

    [max_Idx{:}] = ind2sub(size(samples),max_idx);

    % check if maximum order has been reached
    add_itpl = num_nodes < options.max_nodes;
    % add_itpl = cellfun(@(ip,mi)length(ip)<mi,nodes_part,num2cell(options.max_nodes));
    if ~any(add_itpl)
        fprintf('Reached maximum number of interpolation points \n')
        break
    end

    % add interpolation points
    for i = 1:num_vars
        % make sure to keep at least one sample in LS partition
        if add_itpl(i)
            nodes_part{i} = unique([nodes_part{i},max_Idx{i}]);
            % also interpolate the complex conjugate if 'real_loewner=true'
            if i == 1 && options.itpl_cc && imag(sampling_values{i}(max_Idx{i})) ~= 0
                % find the complex conjugate
                [~,min_idx] = min(abs(conj(sampling_values{1}(max_Idx{i})) - sampling_values{1}));
                nodes_part{i} = unique([nodes_part{i},min_idx]);
            end
        end
    end

    % update number of interpolation points
    num_nodes = cellfun(@length,nodes_part);

    % in order to compute a real loewner matrix we need orthogonal transformation matrices
    if options.real_loewner
        real_transforms.UR = get_JH(sampling_values,nodes_part);
    end

    % solve LS problem
    L = loewner_mat(samples,sampling_values,nodes_part,real_transforms);
    [~,~,X] = svd(L,0);
    denom_coefs = X(:,end);
    
    % set zero coefficients to machine epsilon to avoid numerical issues
    denom_coefs(denom_coefs == 0) = eps;

    rel_lin_ls_err = norm(L * denom_coefs)^2 / norm_2_samples;

    if options.real_loewner
        % transform the real coefficients back to correct complex ones
        denom_coefs = reshape(denom_coefs,flip(num_nodes));
        denom_coefs = tensorprod(real_transforms.UR,denom_coefs,2,num_vars);
        denom_coefs = permute(denom_coefs, [2:num_vars,1]);
        denom_coefs = denom_coefs(:);
    end

    itpl_samples = samples(nodes_part{:});
    if num_vars > 1
        itpl_samples = permute(itpl_samples,num_vars:-1:1);
    end
    itpl_samples = itpl_samples(:);
    nodes = cellfun(@(sv, lp) sv(lp), sampling_values, nodes_part, 'UniformOutput', false);
    num_coefs = denom_coefs .* itpl_samples(:);

    bf = BarycentricForm(nodes,num_coefs,denom_coefs);

    % greedy selection
    err_mat = abs(samples-bf.eval(sampling_values));    

    % set errors to zero to avoid no interpolation points to be added
    zero_idx = cell(1,num_vars);
    for i = 1:num_vars
        if length(nodes_part{i}) >= options.max_nodes(i)
            zero_idx{i} = 1:size(samples,i);
        else
            zero_idx{i} = nodes_part{i};
        end
    end
    err_mat_greedy = err_mat;
    err_mat_greedy(zero_idx{:}) = 0;

    % maximum error for information
    [max_err,~] = max(err_mat,[],'all');

    % maximum error for greedy
    [~,max_idx] = max(err_mat_greedy,[],'all');

    % LS error
    rel_ls_err = norm(err_mat(:))^2 / norm_2_samples;

    info.rel_max_errors(end+1) = max_err / max_samples;
    info.rel_ls_errors(end+1) = rel_ls_err;
    info.rel_linearized_ls_errors(end+1) = rel_lin_ls_err;
    if options.more_info
        info.bf_iterates{end+1} = bf;
    end

    if isfield(options,'validation')
        validation_err_mat = abs(options.validation.samples - bf.eval(options.validation.sampling_values));
        info.rel_validation_max_errors(end+1) = max(validation_err_mat,[],'all') / max(abs(options.validation.samples),[],'all');
        info.rel_validation_ls_errors(end+1) = norm(validation_err_mat(:))^2 / norm(options.validation.samples(:))^2;
    end

    fprintf('p-AAA Iteration %3d | rel max err %.3e | rel LS err %.3e | rel lin LS err %.3e | num nodes [%s]\n', ...
    j, max_err/max_samples, rel_ls_err, rel_lin_ls_err, sprintf('%g ', num_nodes));

end
end

