% with support for multiple streams
function [X Y ValidX ValidY] = default_nn_callback()

global input;
global D;
global Dy;
global nSamples;
global nBatches;
global nValidBatches;

X = cell(1,numel(input.X));
ValidX = cell(1,numel(input.X));
for i = 1:numel(input.X)
   X{i} = single(batchdata_reshape( input.X{i}, [nSamples D{i} nBatches]));
   ValidX{i} = single(batchdata_reshape( input.ValidX{i}, [nSamples D{i} nValidBatches]));
end

Y = cell(1,numel(input.Y));
ValidY = cell(1,numel(input.Y));
for i = 1:numel(input.Y)
   Y{i} = single(batchdata_reshape( input.Y{i}, [nSamples Dy{i} nBatches]));
   ValidY{i} = single(batchdata_reshape( input.ValidY{i}, [nSamples Dy{i} nValidBatches]));
end

