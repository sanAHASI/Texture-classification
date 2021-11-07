imds = imageDatastore('database','IncludeSubfolders',1,...
    'LabelSource','foldernames');
imds.ReadFcn = @readFunctionTraindense;

tbl = countEachLabel(imds);
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8);

net = densenet201;
% Notice the last layer performs 1000 object classification
net.Layers(1)




inputSize = net.Layers(1).InputSize;



if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

 [learnableLayer,classLayer] = findLayersToReplace(lgraph);

% learnableLayer='nnet.cnn.layer.FullyConnectedLayer';

numClasses = numel(categories(imdsTrain.Labels));
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);


newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);


augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);



options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');


densenetgr = trainNetwork(augimdsTrain,lgraph,options);

save densenetgr  densenetgr ;
% load densenet
% [YPred,probs] = classify(densenet,augimdsValidation);
% accuracy = mean(YPred == imdsValidation.Labels);
% idx = randperm(numel(imdsValidation.Files),4);
% figure
% for i = 1:4
%     subplot(2,2,i)
%     I = readimage(imdsValidation,idx(i));
%     imshow(I)
%     label = YPred(idx(i));
%     title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
% end


