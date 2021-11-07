% Copyright 2017 The MathWorks, Inc.

function I = readFunctionTraindense(filename)
% Resize the flowers images to the size required by the network.
I = imread(filename);
 if ismatrix(I)
            I = cat(3,I,I,I);
        end
I = imresize(I, [224 224]);
