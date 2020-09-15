function y = Pool(x)
% The pool function, takes 1 input, and it use MATLAB built-in 'Conv2'
% function to perform convolution 

[xrow, xcol, numFilters] = size(x);
y = zeros(fix(xrow/2), fix(xcol/2), numFilters);

for k = 1:numFilters
    filter = ones(2) / (2*2);
    image = conv2(x(:, :, k), filter, 'valid');
    y(:, :, k) = image(1:2:end, 1:2:end);
end
end

