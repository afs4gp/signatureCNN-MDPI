function y = Conv(x, W)
% This function takes two input - x and W
% We are not implementing the convolution function in code
% MATLAB has already a built-in function to perform convolution operation
% called 'Conv2' 
[wrow, wcol, numFilters] = size(W);
[xrow, xcol, ~         ] = size(x);

yrow = xrow - wrow + 1;
ycol = xcol - wcol + 1;

y = zeros(yrow, ycol, numFilters);

for k = 1:numFilters
    filter = W(:, :, k);
    filter = rot90(squeeze(filter), 2);
    y(:, :, k) = conv2(x, filter, 'valid');
end
end