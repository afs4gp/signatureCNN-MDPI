loadedImages = load('SignaturesRandomized.mat', 'randSignatures'); %Loads the dataset
Images = cell2mat(struct2cell(loadedImages)); %Used to extract the matrix variable from the struct variable that was loaded by load() function above
Images = reshape(Images, 86, 200, []); %Takes the single matrix dataset and reshapes it into a set of 2D images of dimesions 86 rows (height) and 200 columns (width)

Labels = load('SignatureLabels.mat', 'Labels'); %Load the labels from labels file
Labels = cell2mat(struct2cell(Labels)); %Extract the matrix variable from the struct variable that was loaded

convlWeights = 1e-1*randn([3 3 20]); %Weight matrix of the convolution filter (20 filters, each of size 3x3), initialized with random values
fclWeights = (2*rand(100, 83160) - 1) * sqrt(2) / sqrt(100 + 83160); %Initialize the weights used between the input layer of the fully connected layer and its hidden layer
outlWeights = (2*rand( 4,  100) - 1) * sqrt(2) / sqrt( 4 +  100); %Initialize the weights used between the hidden layer and the output layer

trainImages = Images(:, :, 1:80); %Extract the first 80 signatures images for training
trainLabels = Labels(1:80); %Extract the corresponding labels of the extracted images

%TRAINING
for Epoch = 1:10
    Epoch %Displays current training epoch in MATLAB command window
    
    %Pass the weights and the dataset to the TrainCNN function. The
    %training function returns new adjusted weights that will be used to testing
    [convlWeights, fclWeights, outlWeights] = TrainCNN(convlWeights, fclWeights, outlWeights, trainImages, trainLabels);
end

%save('TrainCNN.mat'); %save the weight matrices and other values as a
%separate file

%TESTING
testImages = Images(:, :, 81:100);%extract remaining 20 images for testing (86x200x20)
testLabels = Labels(81:100);%extract the corresponding labels (20x1)
correctPredictions = 0;%used to determine the accuracy of the trained network
numSamples = length(testLabels);%number of images used for testing

for k = 1:numSamples
    image = testImages(:, :, k); %extract a single signature image (86x200)
    fltrdImages = Conv(image, convlWeights); %use 20 trained convolution filters and the signature to filter it and return 20 filtered images (84x198x20)
    convlOut = ReLU(fltrdImages); %pass the filtered images through a ReLU activation layer to remove all negative values
    pooledImages = Pool(convlOut); %pass the filtered images with negative values removed to the pooling function. It returns 20 pooled images (42x99x20)
    flattenedLayer = reshape(pooledImages, [], 1); %flatten the 20 images into a single column matrix (83160x1, that would become the input layer of the fcl)
    fclNeurons = fclWeights*flattenedLayer; %do dot multiplication on the flattened layer with trained weights to get a multiplied sum for each neuron in the hidden layer (100 neurons)
    fclOut = ReLU(fclNeurons); %activation function used for the hidden layer neuron summed values
    outputNodes  = outlWeights*fclOut; %do dot multiplication on the hidden layer with trained weights to get a multiplied sum for each output node in the output layer (4 outputs)
    outputs  = Softmax(outputNodes); %activation function used for the output layer nodes summed values
    
    [~, predictedValue] = max(outputs); %check for the maximum value from the 4 outputs which is the predictedValue
    if predictedValue == testLabels(k) %if prediction equals the actual signature author then number of correct predictions is incremented
        correctPredictions = correctPredictions + 1;
    end
end
    accuracy =  (correctPredictions / numSamples)*100;
    fprintf('Accuracy is %f percent\n', accuracy);
    