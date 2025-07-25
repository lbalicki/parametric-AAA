function [bf,info] = lr_paaa(samples,sampling_values,tol,coef_rank,options)
% LR_PAAA Multivariate rational approximation with the low-rank p-AAA algorithm.
%
%   [BF, INFO] = LR_PAAA(SAMPLES, SAMPLING_VALUES, TOL, COEF_RANK, OPTIONS)
%   Computes a multivariate rational approximant in terms of a barycentric form with low-rank tensor for barycentric coefficients.
%
%   Inputs:
%       SAMPLES             - Multidimensional array of size N_1 x ... x N_d containing the samples to be approximated.
%       SAMPLING_VALUES     - Cell array of sampling points in each variable such that size(sampling_values{i},2) == N_i.
%       TOL                 - Convergence tolerance in terms of relative maximum error.
%       COEF_RANK           - Constraint for number of terms included in the CP decomposition used to represent the barycentric coefficients.
%       OPTIONS (Optional)  - Struct containing options:
%                            * options.nodes_part                   - Cell array of initial nodes (default: empty).
%                            * options.max_nodes                    - Maximum number of nodes in each variable (default: size(samples) - 1).
%                            * options.real_loewner                 - Wether to make the Loewner matrix real-valued. This is only supported if the first variable 
%                                                                     is sampled in complex conjugate pairs and all other variables are real-valued (default: false).
%                            * options.max_iter                     - Maximum number of iterations.
%                            * options.reuse_coefs_ALS              - Wether or not to reuse barycentric coefficients as ALS intialization (default: true).
%                            * options.more_info                    - Wether or not to compute more information about the iterates (default: false).
%                            * options.validation.samples           - Samples used to compute a validation error (default: N\A)
%                            * options.validation.sampling_values   - Sampling values used to compute a validation error (default: N\A)
%                            * options.als_options                  - Struct with additional options for solve_als.
%
%   Outputs:
%       BF              - Rational approximant as a BarycentricForm instance.
%       INFO            - Cell array with information about the approximation at each iteration.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Initialize default options and prepare iteration.
%

if nargin < 5
    options = struct;
end

num_vars = length(sampling_values);

if ~isfield(options,'nodes_part')
    nodes_part = cell(1,num_vars);
else
    nodes_part = options.nodes_part;
end

if ~isfield(options,'als_options')
    options.als_options = struct;
end

if ~isfield(options,'real_loewner')
    options.real_loewner = false;
end

if ~isfield(options,'reuse_coefs_ALS')
    options.reuse_coefs_ALS = true;
end

if ~isfield(options,'max_nodes')
    options.max_nodes = size(samples) - 1;
    if options.real_loewner
        options.max_nodes(1) = options.max_nodes(1) - 1;
    end
end

if ~isfield(options,'max_iter')
    options.max_iter = prod(cellfun(@length,sampling_values))-1;
end

if ~isfield(options,'more_info')
    options.more_info = false;
end

options.als_options.more_info = options.more_info;

% in order to compute a real loewner matrix we need orthogonal transformation matrices  
% and must ensure that data appears in complex conjugate pairs
if options.real_loewner
    [options.als_options.real_transforms.UL,sampling_values,samples] = init_JD(sampling_values,samples);
end

% arrays for storing information about iterates and errors
info.bf_iterates = {};
info.als_info = {};
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
fprintf('LR p-AAA Initial     | rel max err %.3e | rel LS err %.3e\n', max_err/max_samples, rel_ls_err);

% do this such that p-AAA does at least one iteration
max_err = Inf;

max_Idx = cell(1,num_vars);
j = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Main low-rank p-AAA iteration
%

while max_err/max_samples > tol && j < options.max_iter

    j = j + 1;

    [max_Idx{:}] = ind2sub(size(samples),max_idx);

    % check if maximum order has been reached
    add_itpl = cellfun(@(ip,mi)length(ip)<mi,nodes_part,num2cell(options.max_nodes));
    if ~any(add_itpl)
        fprintf('Reached maximum number of interpolation points \n')
        break
    end

    % add interpolation points
    for i = 1:num_vars
        % make sure to keep at least one sample in LS partition
        if add_itpl(i)
            nodes_part{i} = unique([nodes_part{i},max_Idx{i}],'stable');
            % also interpolate the complex conjugate in the first variable if 'real_loewner=true'
            if i == 1 && options.real_loewner && imag(sampling_values{i}(max_Idx{i})) ~= 0
                % find the complex conjugate
                [~,min_idx] = min(abs(conj(sampling_values{1}(max_Idx{i})) - sampling_values{1}));
                nodes_part{i} = unique([nodes_part{i},min_idx],'stable');
            end
        end
    end

    % in order to compute a real loewner matrix we need orthogonal transformation matrices
    if options.real_loewner
        options.als_options.real_transforms.UR = get_JH(sampling_values,nodes_part);
    end

    % if we reuse barycentric coefficients as the ALS initialization we must extract them from the previous barycentric form
    if options.reuse_coefs_ALS
        if j == 1
            coefs_init = cellfun(@(ip) randn(length(ip), coef_rank), nodes_part, 'UniformOutput', false);
        else
            prev_coef_rank = length(bf.denom_coefs.lambda);
            prev_coefs = bf.denom_coefs.U;
            coefs_init = cell(1,num_vars);
            for i = 1:num_vars
                coefs_init{i} = zeros(length(nodes_part{i}), coef_rank);
                coefs_init{i}(1:size(prev_coefs{i},1),1:size(prev_coefs{i},2)) = prev_coefs{i};
                % if the previous coefficients were rank-deficient need to fill
                % up with random values
                if prev_coef_rank < coef_rank
                    coefs_init{i}(:,prev_coef_rank+1:end) = rand(length(nodes_part{i}), coef_rank - prev_coef_rank);
                end
            end
        end
        options.als_options.coefs_init = coefs_init;
    end

    % solve LS problem via ALS  
    [bf,new_als_info] = solve_als(samples,sampling_values,nodes_part,coef_rank,options.als_options);

    info.rel_linearized_ls_errors(end+1) = new_als_info.als_rel_linearized_ls_errors(end);

    % append new information about iterates
    info.als_info{end+1} = new_als_info;

    % greedy selection
    err_mat = abs(samples-bf.eval(sampling_values));

    % set certain errors to zero to avoid no interpolation points to be added
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

    % maximum error for output
    [max_err,~] = max(err_mat,[],'all');

    % maximum error for greedy
    [~,max_idx] = max(err_mat_greedy,[],'all');

    % relative LS error
    rel_ls_err = norm(err_mat(:))^2 / norm_2_samples;

    info.rel_max_errors(end+1) = max_err / max_samples;
    info.rel_ls_errors(end+1) = rel_ls_err;

    % append the barycentric forms of each iteration if more information is desired
    if options.more_info
        info.bf_iterates{end+1} = bf;
    end

    % if validation data was passed compute validation errors
    if isfield(options,'validation')
        validation_err_mat = abs(options.validation.samples - bf.eval(options.validation.sampling_values));
        info.rel_validation_max_errors(end+1) = max(validation_err_mat,[],'all') / max(abs(options.validation.samples),[],'all');
        info.rel_validation_ls_errors(end+1) = norm(validation_err_mat(:))^2 / norm(options.validation.samples(:))^2;
    end

    fprintf('LR p-AAA Iter    %3d | rel max err %.3e | rel LS err %.3e | num nodes [%s]\n', ...
    j, max_err/max_samples, rel_ls_err, sprintf('%g ', cellfun(@length, nodes_part)));

end

end