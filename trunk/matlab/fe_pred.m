%{
Copyright (C) 2013 Yichuan Tang. 
contact: tang at cs.toronto.edu
http://www.cs.toronto.edu/~tang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
%}

rg;
clear all;

% parameters
TESTDATA_FILE = 'test.mat';
MODEL_FILE = 'model5.31.mat';

gg;

% load data
load(TESTDATA_FILE);
models = load(MODEL_FILE);

bad_ind = find(sum(TestX.^2,2) < 1e-2);
TestX(bad_ind,:) = repmat(mean(TestX,1), length(bad_ind),1);

%%
if 1
    TestX = ncc( TestX, models.hp.CC);
elseif 0
    TestX = models.hp.CC*bsxfun(@rdivide, TestX, sqrt(sum(TestX.^2,2)));  
end

nClasses = 7;
nTest = size(TestX,1);
nJit = size(TestX,3);

global D;
global Dy;

D =[];
Dy =[];
D{1} = models.hp.D;
Dy{1} = 1;

TestPredY = zeros(nTest, nClasses, length(models.cv_models));

for mm = 1:length(models.cv_models)
    
    fprintf('\n split:%d ', mm);
    
    model = models.cv_models{mm};
    nSamples = models.hp.nSamples;
    nJitterTrials = models.hp.nJitterTrials;
    
    nTest = size(TestX,1)*size(TestX,3);
    nbatches2 = ceil( nTest/nSamples);
    nrepmat2 = nbatches2*nSamples-nTest;
    
    global input;
    input =[];
    input.X{1} = [batchdata_reshape(TestX(:,:,:)); ...
        zeros( nrepmat2, size(TestX,2))];
    
    input.Y{1} = zeros(size(input.X{1},1), Dy{1});
    
    if nJit == 1 && nJitterTrials > 1
        input.X{1} = repmat(input.X{1}, [nJitterTrials, 1]);
        input.Y{1} = repmat(input.Y{1}, [nJitterTrials, 1]);
        
    elseif nJit > 1
        nJitterTrials = nJit;
    end
        
    model.nSamples = nSamples;
    model.nLayerEst = length(model.net_layers)-1;
    model.MODE = 'classify';
    model.nTestingMode = 1;
    
    [~, res] = myclassify_conv_nn_softmax(model);
    
    
    Yest2 = zeros(size(TestX,1), 7);
    
    for kk = 1:length(res.y_est)
        Yest = double(batchdata_reshape(res.y_est{kk}));
        
        Yest = batchdata_reshape(Yest, ...
            [size(Yest,1)/nJitterTrials, 7, nJitterTrials]);
        Yest = Yest(1:size(TestX,1),:,:);
        
        for k = 1:nJitterTrials,
            temp = double(squeeze(Yest(:,:,k)));
            if model.net_layers{end}.nNeuronType ==6
                Z = logsumexp(temp,2);
                temp = exp( bsxfun(@minus, temp, Z));
            end
            Yest2 = Yest2 + temp;
        end
    end
    
    Yest2 = Yest2./(length(res.y_est)*nJitterTrials);        
    TestPredY(:,:,mm) =Yest2;    
end

TestPredY2 = TestPredY;
TestPredY2 = mean(TestPredY2,3);
[val ind] = max(TestPredY2, [], 2);

%%
if 1
    filename = 'sample_submission.csv';
    fid = fopen(filename, 'w+');

    for i = 1:size(TestPredY2,1)
        fprintf(fid,'%d\n', int32(ind(i)-1));
    end

    fprintf('\n Finished writing to %s\n', filename);
    fclose(fid);
end
