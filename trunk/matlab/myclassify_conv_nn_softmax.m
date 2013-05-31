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

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CT 12/2012
derived from myclassify_nn_softmax.m
PURPOSE: to do regression using a neural net
INPUT:
OUTPUT:
NOTES:
requires these global variables:
global nSamples;
global D;
global Dy;
global nBatches;
global nValidBatches;

TESTED:
CHANGELOG:
TODO:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function [model res] = myclassify_conv_nn_softmax( model )

global input;
global D;
global Dy;

assert(model.nDataSources == length(input.X));
assert(model.nObjectiveSinks == length(input.Y));


for i = 1:length(input.X)
    input.X{i} = batchdata_reshape(input.X{i});
end

for j = 1:length(input.Y)
     input.Y{j} = batchdata_reshape(input.Y{j});
end

if strcmp( model.MODE, 'fit')
    
    assert(model.nDataSources == length(input.ValidX));
    assert(model.nObjectiveSinks == length(input.ValidY));  
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % standardize data
    for i = 1:length(input.X)
      
        if model.RM_MEAN_STD{i} ==1
            model.X_mean{i} = mean([input.X{i}; input.ValidX{i}], 1);
            model.X_std{i} = sqrt(var( [input.X{i}; input.ValidX{i}]) + model.min_data_std{i}^2);

            input.X{i} = bsxfun(@rdivide, bsxfun(@minus, input.X{i}, model.X_mean{i}), model.X_std{i});
            input.ValidX{i} = bsxfun(@rdivide, bsxfun(@minus, input.ValidX{i}, model.X_mean{i}), model.X_std{i});

        elseif model.RM_MEAN_STD{i} ==2 % PCA
            [~, model.T{i}] = cl_whiten( double([input.X{i}; input.ValidX{i};]) );
            
            model.T{i}.sqrtD = model.T{i}.sqrtD(end-(model.PCA_D{i}-1):end);
            model.T{i}.invsqrtD = model.T{i}.invsqrtD(end-(model.PCA_D{i}-1):end);
            model.T{i}.Ut = model.T{i}.Ut(end-(model.PCA_D{i}-1):end,:);

            input.X{i} = cl_whiten_fwd(  input.X{i}, model.T{i} );
            input.ValidX{i} = cl_whiten_fwd(input.ValidX{i}, model.T{i});
        
        elseif model.RM_MEAN_STD{i} == 20 % PCA set externally            
            input.X{i} = cl_whiten_fwd(  input.X{i}, model.T{i} );
            input.ValidX{i} = cl_whiten_fwd(input.ValidX{i}, model.T{i});            
        else
            model.X_mean{i} =[];
            model.X_std{i} = [];
        end        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    params = [];    
    params.maxepoch = length(model.rates);
    
    params.momen = single(model.momen);
    params.rates = single(model.rates);    
    params.noise = single(model.noise);
    params.adagrad = single(model.adagrad);
    params.save_wts_epochs = single(model.save_wts_epochs);    
    
    params.nSamples = model.nSamples;
    params.nValidBatches = model.nValidBatches;
    params.nBatches = model.nBatches;
    params.BEST = model.BEST;
    params.bRefill = model.bRefill;
    
    params.nDataSources = model.nDataSources;
    params.nObjectiveSinks = model.nObjectiveSinks;
    params.net_layers = model.net_layers;
    params.nEpochsLookValid = model.nEpochsLookValid;    
    
                
    if model.INIT_W==1
        %%%%%%%%%%%%%%%%%%%%%  network initialization %%%%%%%%%%%%%%%%%%%%%%%%%         
        [ net ] = net_layers_init( model.net_layers, model.nDataSources );
        if isfield(model, 'net_wt_shares'),
            params.net_wt_shares = model.net_wt_shares;

            for ii = 1:length(params.net_wt_shares)            
                assert(length(params.net_wt_shares{ii}) > 1);
                layer_ref = params.net_wt_shares{ii}(1);
                assert( strcmp( model.net_layers{layer_ref}.type, 'fc'));

                for jj = 2:length(params.net_wt_shares{ii})                 
                    layer_2 = params.net_wt_shares{ii}(jj);
                    assert(strcmp( model.net_layers{layer_2}.type, 'fc'));                
                    net.W{layer_2-model.nDataSources} = net.W{layer_ref-model.nDataSources}';
                end            
            end
        end

        [ww] = net_layers_serialize( net, model.net_layers, model.nDataSources );    
        [n2] = net_layers_deserialize( ww, model.net_layers, model.nDataSources );
        [ww2] = net_layers_serialize( n2, model.net_layers, model.nDataSources );
        assert( maxabs(ww,ww2) == 0);
        
    else
        assert(length(model.theta) ==1);
        ww = model.theta{1};
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % assertions
    assert(length(params.momen)==length(params.rates));
    assert(length(params.noise)==length(params.rates));
    
    for i = 1:length(input.X)
        [n d nBatches] = size(input.X{i});
        assert(d == D{i});
        assert(n*nBatches >= model.nBatches*model.nSamples);        
    end
    
    for j = 1:length(input.Y)
        [n d nBatches] = size(input.Y{j});
        assert(d == Dy{j});
        assert(n*nBatches >= model.nBatches*model.nSamples);
        
        if model.net_layers{end-model.nObjectiveSinks+j}.nNeuronType == 5              
            d_output = max(input.Y{j}(:))+1;
            assert(model.net_layers{end-model.nObjectiveSinks+j}.nH == d_output);
            assert( min(input.Y{j}(:)) == 0 );
            assert(d == 1);
        end        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
    tic;
    if model.USECPU == 0
        
        params.CHECKGRAD = 0;        
        [~, model.theta] = mexcuConvNNoo( single(ww), params, model.callback_name);    
 
    end
    fprintf('\n');
    toc;
       
    
elseif strcmp( model.MODE, 'classify')
  
  
    assert(numel(size(input.X))==2);
    assert(numel(size(input.Y))==2);
    
    
    xtemp = cell(1, length(input.X));
    ytemp = cell(1, length(input.Y));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % standardize data
    for i = 1:length(input.X)
        if model.RM_MEAN_STD{i} ==1
            input.X{i} = bsxfun(@rdivide, bsxfun(@minus, input.X{i}, model.X_mean{i}), model.X_std{i});
        elseif model.RM_MEAN_STD{i} ==2 || model.RM_MEAN_STD{i} == 20   % PCA
            input.X{i} = cl_whiten_fwd(input.X{i}, model.T{i} );
        else
            %do nothing
        end
       
        assert( rem(size(input.X{i},1),model.nSamples) == 0);
        nb = size(input.X{i},1)/model.nSamples;
        if nb > 1
            xtemp{i} = single(batchdata_reshape( input.X{i}, [model.nSamples D{i} nb]));
        else
            xtemp{i} = single(input.X{i});
        end
      
    end
    
    for j = 1:length(input.Y)
        assert( rem(size(input.Y{j},1), model.nSamples) == 0);
        nb = size(input.Y{j},1)/model.nSamples;
        if nb > 1
            ytemp{j} = single(batchdata_reshape( input.Y{j}, [model.nSamples Dy{j} nb])); 
        else
            ytemp{j} = single(input.Y{j});
        end
    end
    
    
    params =[];   
    params.nSamples = model.nSamples;
    params.nValidBatches = nb; %mexcuConvNNooFF uses nValidBatches    
    params.nDataSources = model.nDataSources;
    params.nObjectiveSinks = model.nObjectiveSinks;
    params.net_layers = model.net_layers;
    params.nLayerEst =  model.nLayerEst;
    params.nTestingMode = model.nTestingMode;
    
    params.nVerbose = 0;    
    res = [];
    res.y_est=cell(length(model.theta), model.nObjectiveSinks);
    for t = 1:length(model.theta)
        [loss, y_est] = mexcuConvNNooFF( single(model.theta{t}), params, xtemp, ytemp);
        res.y_est(t,:) = y_est;
    end
    
    fprintf('\nConvNNSoftmax::classify \tLoss:%e\n', loss);
    res.loss = loss;
    
else
    assert(false);
end
