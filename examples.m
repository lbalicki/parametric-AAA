
%% 2-variable scalar function

% Generate test data
x = linspace(-5,5,100);
y = linspace(-5,5,100);
samples = (x.' + y) ./ (4 + cos(x.') + cos(y));

% Approximate data via p-AAA with error tolerance 1e-5
bf = paaa(samples,{x,y},1e-5);

% Evaluate rational approximation at a single points
bf.eval([0,0])

% Evaluate rational approximation at multiple points
bf.eval([0,0;1,1;2,2])

% Evaluate rational approximation on a grid
bf.eval({[1,2,3],[4,5,6]})

% Compute the poles with respect to the first variable x, and the fixed value y=1
bf.poles({1},1)

% Compute the poles with respect to the second variable y, and the fixed value x=5
bf.poles({5},2)

% Compute multiple poles with respect to the second variable y, and the fixed values x=1,3,5
bf.poles({[1,3,5]},2)

%% 2-variable matrix-valued function

% Generate test data
x = linspace(-5,5,100);
y = linspace(-5,5,100);
samples = zeros(100,100,2,2);
for i = 1:100
    for j = 1:100
        samples(i,j,:,:) = [x(i)*y(j), x(i)^2; y(j)^2+1, 1] / (4 + cos(x(i)) + cos(y(j)));
    end
end

% Approximate data via set-valued p-AAA with error tolerance 1e-5
block_bf = sv_paaa(samples,{x,y},1e-5);

% Evaluate rational approximation at a single points
block_bf.eval([0,0])

% Evaluate rational approximation on a grid
block_bf.eval({[1,2,3],[4,5,6]})

%% 3-variable scalar function

% Generate test data
x = linspace(-5,5,25);
y = linspace(-5,5,25);
z = linspace(-5,5,25);
samples = zeros(25,25,25);
for i = 1:25
    for j = 1:25
        for k = 1:25
            samples(i,j,k) = (x(i) + y(j) + z(k)) ./ (6 + cos(x(i)) + cos(y(j)) + cos(z(k)));
        end
    end
end

% Approximate data via low-rank p-AAA with error tolerance 1e-3 and tensor rank 3
lr_bf = lr_paaa(samples,{x,y,z},1e-3,3);
