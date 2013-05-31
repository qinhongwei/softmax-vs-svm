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

function [ cv_average, cv_models ] = fe_cv_48( ...
          nSPLIT, randseeds, normalseeds, hp)
global DataX;
global DataY;

rand('seed', randseeds);
nTrain = size(DataX,1);
nJit = size(DataX,3);

assert(size(DataY,1)==nTrain);

assert(rem(nTrain,nSPLIT) == 0);
nTrainPerSplit = nTrain/nSPLIT;

inds = reshape(randperm(nTrain), nSPLIT, nTrainPerSplit);

DataXSplit =cell(nSPLIT,1);
DataYSplit =cell(nSPLIT,1);
 
for ii = 1:nSPLIT
    DataXSplit{ii} = DataX(inds(ii,:),:,:);
    DataYSplit{ii} = DataY(inds(ii,:),:,:);
   
end

cv_models = [];

hp.the_splits
for nn = hp.the_splits
    
    fprintf('\nSplit:%d host:\n', nn);
    system('hostname');
    
    %% training
    randn('seed', normalseeds);
    %rand('seed', randseeds+nn);
    
    global nSamples;
    global D;
    global Dy;
    global nBatches;
    global nValidBatches;
    
    nDataSources = 1;
    nObjectiveSinks = 1;
    
    nSamples = hp.nSamples;
    D =[];
    Dy =[];
    D{1} = hp.D;
    Dy{1} = 1;     
    
    traincell = DataXSplit(setdiff(1:nSPLIT, nn));
    tempdata = cat(1, traincell{:});    
    
    traincellY = DataYSplit(setdiff(1:nSPLIT, nn));
    tempdataY = cat(1, traincellY{:});
        
    tempdata = [batchdata_reshape(tempdata)];
    tempdataY = [batchdata_reshape(tempdataY)];    
        
    nBatches = floor(size(tempdata,1)/nSamples);
    nValidBatches = nJit*nTrainPerSplit/nSamples;
    
    global input;
    input =[];
    traininds = randperm(size(tempdata,1));
    input.X{1} = tempdata(traininds(1:nSamples*nBatches),:);
    input.Y{1} = tempdataY(traininds(1:nSamples*nBatches),:);
        
    input.ValidX{1} = batchdata_reshape(DataXSplit{nn}(:,:,:));
    input.ValidY{1} = batchdata_reshape(DataYSplit{nn}(:,:,:));
        
    model = [];
    model.net_layers = hp.net_layers; 
    
    model.RM_MEAN_STD{1} = hp.RM_MEAN_STD;
    model.PCA_D{1} = 70;
    model.min_data_std{1} = 0.01;
        
    if model.RM_MEAN_STD{1} == 2
        D{1} = model.PCA_D{1};
    end
    
    model.RM_MEAN_STD{2} = 0;
    
    %-------------------
    model.MaxIters =  hp.MaxIters;
    start_rate = hp.start_rate;
    HalfLife = hp.HalfLife;
    model.rates = start_rate*ones(1,model.MaxIters);
    model.rates = model.rates./(1+[1:model.MaxIters]/HalfLife);
    model.rates(int32(end-model.MaxIters/3):end) = ...
        model.rates(int32(end-model.MaxIters/3):end)/10;
    
    model.rates(int32(end-model.MaxIters/6):end) = ...
        model.rates(int32(end-model.MaxIters/6):end)/10;
    
    model.noise = linspace(hp.noise1, hp.noise2, model.MaxIters);
    
    model.momen = hp.momen*ones(1, model.MaxIters);
    model.momen(1:hp.momen_init) = 0.5;
        
    model.adagrad = 0*ones(1, model.MaxIters);
    
    saveepochs = linspace(0.90, .999, 5);
    model.save_wts_epochs = floor(saveepochs*model.MaxIters);
    
    model.nSamples = nSamples; %was at 500
    model.nBatches = nBatches;
    model.nValidBatches = nValidBatches;
    model.nDataSources = nDataSources;
    model.nObjectiveSinks = nObjectiveSinks;
    model.nEpochsLookValid = 2000;
    
    model.USECPU = 0;
    model.callback_name = 'default_nn_callback';
    model.BEST = 0;
    model.bRefill = 0;
    model.MODE = 'fit';
    model.INIT_W = 1;
      
    [model] = myclassify_conv_nn_softmax(model);
    
    cv_models{nn} = model;
    
    
    %% validation
       
    input = [];
    input.X{1} = batchdata_reshape(DataXSplit{nn}(:,:,:));
    input.Y{1} = batchdata_reshape(DataYSplit{nn}(:,:,:));
    
    nJitterTrials = hp.nJitterTrials;
    if nJit == 1 && nJitterTrials > 1
       input.X{1} = repmat(input.X{1}, [nJitterTrials, 1]);
       input.Y{1} = repmat(input.Y{1}, [nJitterTrials, 1]);
       
    elseif nJit > 1
        nJitterTrials = nJit;
    end
    nJitterTrials
    
    model.nSamples = nSamples;
    model.nLayerEst = length(model.net_layers)-1;
    model.MODE = 'classify';
    model.nTestingMode = 1;
    
    [~, res] = myclassify_conv_nn_softmax(model);
    
    nClasses =7;  
    Yest2 = zeros(size(DataYSplit{nn},1), nClasses);
     
    for kk = 1:length(res.y_est)
        Yest = double(batchdata_reshape(res.y_est{kk}));

        Yest = batchdata_reshape(Yest, ...
                [size(DataXSplit{nn},1), nClasses, nJitterTrials]);

        Yest_kk = zeros( size(input.Y{1},1)/nJitterTrials, nClasses);
        for k = 1:nJitterTrials,
            temp = squeeze(Yest(:,:,k));
            if model.net_layers{end}.nNeuronType ==6
                Z = logsumexp(temp,2);
                temp = exp( bsxfun(@minus, temp, Z));
            end
            Yest_kk = Yest_kk + temp;
        end    
        Yest2 = Yest2 + Yest_kk;
    end
    Yest2 = Yest2./length(res.y_est)/nJitterTrials;    
     
    [val ind] = max(Yest2,[],2);
    cv_average(nn) = mean(ind==DataYSplit{nn}(:,:,1)+1);
    cv_average
    
    
end %nSplits

dummy = 1;
