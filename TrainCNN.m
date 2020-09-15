function [convlWeights, fclWeights, outlWeights] = TrainCNN(convlWeights, fclWeights, outlWeights, trainImages, trainLabels)

% This function train our Convolutional Neural Network

alpha = 0.01; % learning rate
beta = 0.95;

momentumConvl = zeros(size(convlWeights));
momentumFcl = zeros(size(fclWeights));
momentumOutl = zeros(size(outlWeights));

N = length(trainLabels);

batchSize = 10;
batchList = 1:batchSize:(N-batchSize+1); % 1:80

for batch = 1:length(batchList)
    dconvlWeights = zeros(size(convlWeights));
    dfclWeights = zeros(size(fclWeights));
    doutlWeights = zeros(size(outlWeights));

    begin = batchList(batch);
    for k = begin:begin+batchSize-1

        image = trainImages(:, :, k);
        fltrdImages = Conv(image, convlWeights);
        convlOut = ReLU(fltrdImages);
        pooledImages = Pool(convlOut);
        flattenedLayer = reshape(pooledImages, [], 1);
        fclNeurons = fclWeights*flattenedLayer;%it is the dot product of W5 matrix and y4 matrix, v5 is the neurons in the hidden layer (100 neurons)
        fclOut = ReLU(fclNeurons);%each neuron has an output which is passed through the relu activation function
        outputNodes  = outlWeights*fclOut;
        outputs  = Softmax(outputNodes);

        d = zeros(4, 1);
        d(sub2ind(size(d), trainLabels(k), 1)) = 1;%this is the right one, this is the expected/correct result

        %from here is the error calculation between output layer and hidden
        %layer
        e      = d - outputs;
        delta  = e;
        e5     = outlWeights' * delta;
        delta5 = (fclOut > 0) .* e5;

        %From here is the error calculation between hidden layer and the
        %flattened layer
        e4     = fclWeights' * delta5;
        e3     = reshape(e4, size(pooledImages)); %error matrix carried on to the pooling layer, between flattened and pooling layers
        e2     = zeros(size(convlOut));
        W3     = ones(size(convlOut)) / (2*2);

        for count = 1:20
            e2(:, :, count) = kron(e3(:, :, count), ones([2 2])) .* W3(:, :, count);
        end

        delta2 = (convlOut > 0) .* e2;

        delta1_x = zeros(size(convlWeights));


        for count = 1:20
            delta1_x(:, :, count) = conv2(image(:, :), rot90(delta2(:, :, count), 2), 'valid');
        end

        dconvlWeights = dconvlWeights + delta1_x;
        dfclWeights = dfclWeights + delta5*flattenedLayer';
        doutlWeights = doutlWeights + delta *fclOut';

    end

    dconvlWeights = dconvlWeights / batchSize;
    dfclWeights = dfclWeights / batchSize;
    doutlWeights = doutlWeights / batchSize;

    momentumConvl = alpha*dconvlWeights + beta*momentumConvl;
    convlWeights = convlWeights + momentumConvl;

    momentumFcl = alpha*dfclWeights + beta*momentumFcl;
    fclWeights = fclWeights + momentumFcl;

    momentumOutl = alpha*doutlWeights + beta*momentumOutl;
    outlWeights = outlWeights + momentumOutl;

end
end
        
    