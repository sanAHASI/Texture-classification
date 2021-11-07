% load ('trainedNet.mat');
% load('densenet.mat');
load('densenetgr.mat');
allImages = imageDatastore('database', 'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');
allImages.ReadFcn=@readFunctionTraindense;
numClasses = numel(categories(allImages.Labels));

%% Split data into training and test sets 
[trainingImages, testImages] = splitEachLabel(allImages, 0.30, 'randomize'); 
numtestClasses = numel(categories(testImages.Labels));
disp(numtestClasses);
nooftestimage = numel((testImages.Files));

[predictedLabels,probs2] = classify(densenetgr,testImages);
trueLabels=testImages.Labels;
figure;
confusionchart(trueLabels,predictedLabels);
figure;
plotconfusion(trueLabels,predictedLabels);

