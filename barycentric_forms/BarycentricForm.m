classdef BarycentricForm
    %BARYCENTRICFORM Proper multivariate rational function in barycentric form.

    properties
        itpl_nodes % cell array of interpolation nodes
        num_coefs % numerator coefficient tensor
        denom_coefs % denominator coefficient tensor
        num_vars % number of variables
        orders % orders of the barycentric form in each variable
        rem_sing_tol = 1e-14 % tolerance for removable singularities
    end
    
    methods
        function obj = BarycentricForm(itpl_nodes,num_coefs,denom_coefs,rem_sing_tol)
            %BARYCENTRICFORM Construct a BarycentricForm instance.
            %
            %   Inputs:
            %       ITPL_NODES    - Cell array of interpolation nodes such that itpl_nodes{i} are the interpolation nodes for the i-th variable.
            %       NUM_COEFS     - Tensor/matrix/vector representing the barycentric coefficients of the numerator.
            %       DENOM_COEFS   - Tensor/matrix/vector representing the barycentric coefficients of the denominator.
            %       REM_SING_TOL  - Tolerance for removable singularities.

            obj.itpl_nodes = itpl_nodes;
            obj.num_vars = length(itpl_nodes);
            obj.orders = cellfun(@length,itpl_nodes)-1;
            obj.denom_coefs = obj.init_coefs_tensor(denom_coefs);
            obj.num_coefs = obj.init_coefs_tensor(num_coefs);
            if nargin > 3
                obj.rem_sing_tol = rem_sing_tol;
            end
        end

        function CT_init = init_coefs_tensor(obj,CT)
            %INIT_COEFS_TENSOR Initialize coefficient tensor based on the passed input type.

            if all(arrayfun(@(i)(size(CT,i) == obj.orders(i)+1), 1:obj.num_vars))
                CT_init = CT;
            elseif size(CT,1) == prod(obj.orders+1)
                CT_init = reshape(CT,flip(obj.orders)+1); % case where vectorized coefficients are passed
                CT_init = permute(CT_init,obj.num_vars:-1:1); % need to do rearranging to account for Matlabs Kronecker product order
            else
                error('Coefficients and barycentric node dimensions do not match.')
            end
        end

        function ps = poles(obj,other_args,var_idx)

            % if no variable index provided, assume we want poles with respect to the first variable
            if nargin < 3
                var_idx = 1;
            end

            % cover special case of single-variable function
            if nargin < 2 || obj.num_vars == 1
                if obj.num_vars > 1
                    error('Need to provide arguments for all but one variable for pole computation.')
                end

                [A,E] = bf_compan(obj.denom_coefs, obj.itpl_nodes{1});
                ps = eig(A,E);
                ps(isinf(ps)) = [];
                return
            end

            % assemble the barycentric coefficient evaluated at the fixed variables
            ap = obj.denom_coefs;
            args_offset = 0;
            for j = 1:obj.num_vars
                if j == var_idx
                    args_offset = 1;
                    continue;
                end

                C = cauchy_mat(obj.itpl_nodes{j}.',other_args{j-args_offset}.').';
                ap = tensorprod(C,ap,2,j);
            end
            if obj.num_vars > 1
                ap = permute(ap, obj.num_vars:-1:1);
            end
            sigma = obj.itpl_nodes{var_idx};

            n = obj.orders(var_idx) + 1;

            E = zeros(n);
            E(1:end-1,1) = ones(n-1,1);
            E(1:end-1,2:end) = -eye(n-1);

            A = zeros(n);
            A(1:end-1,1) = sigma(1);
            A(1:end-1,2:end) = -diag(sigma(2:end));

            size_ap = size(ap);
            size_ps = size_ap;
            size_ps(1) = size_ps(1) - 1;
            ps = zeros(size_ps);

            % compute eigenvalues for each input
            for j = 1:prod(size_ap(2:end))
                A(end,:) = ap(:,j);
                p = eig(A,E);
                p(isinf(p)) = [];
                ps(:,j) = p;
            end
        end

        function H = eval(obj,args)
            %EVAL Evaluate baryenctric form.
            if size(args,2) ~= obj.num_vars
                error('Argument has wrong number of variables.')
            end

            N = obj.num_coefs;
            D = obj.denom_coefs;
            if iscell(args)
                for i = 1:obj.num_vars
                    C = cauchy_mat(args{i},obj.itpl_nodes{i},obj.rem_sing_tol);
                    N = tensorprod(C.',N,2,i);
                    D = tensorprod(C.',D,2,i);
                end
                H = N ./ D;
                if obj.num_vars > 1
                    H = permute(H, obj.num_vars:-1:1);
                end
            else
                kr_C = cauchy_mat(args(:,1).',obj.itpl_nodes{1},obj.rem_sing_tol);
                for i = 2:obj.num_vars
                    C = cauchy_mat(args(:,i).',obj.itpl_nodes{i},obj.rem_sing_tol);
                    kr_C = khatri_rao_prod(C,kr_C);
                end
                H = (kr_C.' * N(:)) ./ (kr_C.' * D(:));
            end
        end
    end
end

