# parametric-AAA

This project contains MATLAB implementations for the parametric adaptive Antoulas-Anderson (p-AAA) algorithm. The p-AAA algorithm computes a multivariate rational function that approximates a given data set. Several variants of the algorithm and efficient implementations for barycentric forms of rational functions are implemented.

Basic example for approximating the function $f(x,y) = \frac{x + y}{4 + cos(x) + cos(y)}$:
```
% Generate test data
x = linspace(-5,5,100);
y = linspace(-5,5,100);
samples = (x.' + y) ./ (4 + cos(x.') + cos(y));

% Approximate data via p-AAA with error tolerance 1e-5
bf = paaa(samples,{x,y},1e-5)

% Evaluate rational approximation
bf.eval([0,0])
```

## Dependencies

- MATLAB R2022a or later
- For low-rank p-AAA implementation: [Tensor Toolbox for MATLAB](https://www.tensortoolbox.org/), version 3.4 or later

## Contents

Barycentric forms:
- `BarycentricForm` Implemention of the barycentric form of a multivariate rational function.
- `BlockBarycentricForm` Implemention of the multivariate barycentric form for vector/matrix valued functions.
- `LowRankBarycentricForm` Implementation of the multivariate barycentric form with barycentric coefficients represented by low-rank tensors.

Algorithms:
- `paaa` Standard p-AAA algorithm.
- `sv_paaa` Set-valued p-AAA for vector/matrix valued functions.
- `lr_paaa` Low-rank p-AAA that represents barycentric coefficients in terms of low-rank tensors. 

## License

The code is available under the MIT license (see `LICENSE.md` for details).
