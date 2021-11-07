% load ('trainedNet.mat');
load('densenetgr.mat');
% load('gnet2.mat');
allImages = imageDatastore('database', 'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');
numClasses = numel(categories(allImages.Labels));

%% Split data into training and test sets 
[trainingImages, testImages] = splitEachLabel(allImages, 0.50, 'randomize'); 
numtestClasses = numel(categories(testImages.Labels));
disp(numtestClasses);
nooftestimage = numel((testImages.Files));

TP=0;
TN=0;
FP=0;
FN=0;

accuracy=0;
MSE=0;
predictedLabels=[];
for i=1:nooftestimage
    
    PrevOut.Positive=0;
    PrevOut.Negative= 0;
    PreprocessesImages= readimage( testImages , i );
    Actualclass=testImages.Labels(i);


% %PrevOut.NSR= 0;
% 
%testImages.ReadFcn =  @readFunctionTrainAlex;
%  PreprocessesImages=readAndPreprocessImagematconvnet(PreprocessesImages);
% [YPred,probs] = classify(net,PreprocessesImages);
% 
% 
% OutputName=cellstr(YPred);
% PrevOut = Majorityvoting (OutputName,PrevOut);
% YPred=char(YPred);
% PreprocessesImages=imresize(PreprocessesImages,[224 224]);
% [YPred2,probs2] = classify(gnet2,PreprocessesImages);
% OutputName=cellstr(YPred2);
% PrevOut = Majorityvoting (OutputName,PrevOut);
% 
% YPred2=char(YPred2);
PreprocessesImages=imresize(PreprocessesImages,[224 224]);
I=PreprocessesImages;
if ismatrix(I)
            I = cat(3,I,I,I);
end
      PreprocessesImages=I;  
[YPred3,probs3] = classify(densenetgr,PreprocessesImages);
OutputName=cellstr(YPred3);
PrevOut = Majorityvoting (OutputName,PrevOut);

YPred3=char(YPred3);
prob=probs3;
probsa(i,:)=prob
MV=[ PrevOut.Positive,PrevOut.Negative];
% axes(handles.axes3);
% bar(MV);
% set(gca,'Xticklabel',{'Pos','Neg'});

[c,ind]=max(MV);

    switch ind

        case 1
           
        FinalClass='Positive';
        predictedL='Positive';
        predictedLabels{i}=char(predictedL);

            %set(handles.edit1,'String','Positive');
            if (Actualclass == 'Positive');
                 TN=TN+1;
            else
                FN=FN+1;

            end

        case 2
        %set(handles.edit1,'String','Negative');
        FinalClass='Negative';
        predictedL='Negative';
        predictedLabels{i}=char(predictedL);


            if (Actualclass == 'Negative');
                 TP=TP+1;
            else
                FP=FP+1;

            end

    end
% True positive: Sick people correctly identified as sick
% False positive: Healthy people incorrectly identified as sick
% True negative: Healthy people correctly identified as healthy
% False negative: Sick people incorrectly identified as healthy

    if (FinalClass == Actualclass);
        accuracy=accuracy+1;
        MSE=MSE+0;
    else
        accuracy=accuracy+0;
        MSE=MSE+1;

    end
end

MSE=MSE/nooftestimage;
disp('MSE');
disp(MSE);
disp(accuracy)
predictedLabels=predictedLabels';
predictedLabels=categorical(predictedLabels);
    trueLabels=testImages.Labels;
    figure;
    confusionchart(trueLabels,predictedLabels);
figure;
plotconfusion(trueLabels,predictedLabels);

figure;
YPred=predictedLabels;
testeImagesLabels=trueLabels;
cgt = double(testeImagesLabels); clabel = double(YPred); cscores = double(probsa);
figure;
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(cgt,cscores(:,1),1); 
plot(X,Y,'r');
grid on
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Densenet')

P=50;
N=50;
%Accuracy
Acc=((TP+TN)/(P+N));
disp('Acc');

disp(Acc);
%Sensitivity

TPR=TP/P;
disp('Sensitivity');

disp(TPR);
%SPECIFICTY

TNR=TN/N;
disp('SPECIFICTY');

disp(TNR);
%PRECISION

PPV=TP/(TP+FP);
disp(PPV);

%MISS RATE

FNR=FN/P;
disp(FNR);

%F1 SCORE
disp('F1');

F1=(2*TP)/((2*TP)+FP+FN);
disp(F1);

%BALANCED ACCURACY

BA=(TPR+TNR)/2;

disp(BA);
% MCC
MCC=((TP*TN)-(FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
disp('MCC');

disp(MCC);

%AUC
disp('AUC');

disp(AUC);
