% load ('trainedNet.mat');
% load('densenet.mat');
load('vggnet19.mat');
allImages = imageDatastore('databasecode', 'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');
numClasses = numel(categories(allImages.Labels));

%% Split data into training and test sets 
[trainingImages, testImages] = splitEachLabel(allImages, 0.50, 'randomize'); 
numtestClasses = numel(categories(testImages.Labels));
disp(numtestClasses);
nooftestimage = numel((testImages.Files));



accuracy=0;
MSE=0;
predictedLabels=[];
for i=1:nooftestimage
    
    
    PreprocessesImages= readimage( testImages , i );
    Actualclass=testImages.Labels(i);


PreprocessesImages=imresize(PreprocessesImages,[224 224]);
if ismatrix(PreprocessesImages)
            PreprocessesImages = cat(3,PreprocessesImages,PreprocessesImages,PreprocessesImages);
end
        
[YPred2,probs2] = classify(netmini,PreprocessesImages);
OutputName=cellstr(YPred2);
% PrevOut = Majorityvoting (OutputName,PrevOut);

YPred2=char(YPred2);
        predictedLabels{i}=char(YPred2);



    end


    


predictedLabels=predictedLabels';
predictedLabels=categorical(predictedLabels);
    trueLabels=testImages.Labels;
    figure;
    confusionchart(trueLabels,predictedLabels);
figure;
plotconfusion(trueLabels,predictedLabels);

