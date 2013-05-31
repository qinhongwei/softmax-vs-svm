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
%%%%%%%%%%%%%%%%%%%%%%%%

    net_layers{1}.type = 'fc' or 'conv' 

    fc, 
        net_layers{1}.nV
        net_layers{1}.nH
        net_layers{1}.nNeuronType
        net_layers{1}.f_dropout
        net_layers{1}.f_wtcost
    fcdata:
    convc:
        net_layers{1}.nVisChannels
        net_layers{1}.nFilters
        net_layers{1}.nI_filt
        net_layers{1}.nJ_filt        
        net_layers{1}.nVisI
        net_layers{1}.nVisJ
        net_layers{1}.nNeuronType
        net_layers{1}.f_dropout
        net_layers{1}.f_wtcost
    convs:

    convdata:
 
TESTED:
TODO:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}

function [ net ] = net_layers_init( net_layers, nDataSources )

net.W = cell(1, length(net_layers)-nDataSources);
net.hb = cell(1, length(net_layers)-nDataSources);

for i = 1:nDataSources
    assert( strcmp(net_layers{i}.type, 'convdata') ... 
            || strcmp(net_layers{i}.type, 'fcdata') );
end

for ll = (nDataSources+1):length(net_layers)
    
    Lay = net_layers{ll};
    L = ll-nDataSources;
    
     if strcmp(Lay.type, 'fc')
         if Lay.ri_mode == 0
             net.W{L} = Lay.ri*randn(Lay.nV, Lay.nH);
         elseif Lay.ri_mode == 1
                          
             a = Lay.ri/sqrt(Lay.nV);
             
             net.W{L} = a*randn(Lay.nV, Lay.nH);       %Gaussian             
             %net.W{L} = rand(Lay.nV, Lay.nH)*2*a-a;   %uniform
             
         elseif Lay.ri_mode == 2                        %sparse
             net.W{L} = zeros(Lay.nV, Lay.nH);
             for j = 1:Lay.nH
                 inds = randperm(Lay.nH);
                 K = int32(min(15, 0.5*Lay.nH));
                 net.W{L}( inds(1:K), j) = Lay.ri*randn(K,1);
             end
         else
             assert(false);
         end
         
         if isfield(Lay, 'initB')
             initB = Lay.initB;
         else
             initB = 0;
         end
      
         nH_bias = Lay.nH;
         
         net.hb{L} = initB*ones(1, nH_bias);
        
    elseif strcmp(Lay.type, 'convc')
        
        net.W{L} = Lay.ri*randn(Lay.nFilters, Lay.nJ_filt*Lay.nI_filt*Lay.nVisChannels);
        if isfield(Lay, 'initB')
            initB = Lay.initB;
        else
            initB = 0;
        end        
        net.hb{L} = initB*ones(Lay.nFilters, 1);
    
     elseif strcmp(Lay.type, 'deconvc')
        
        net.W{L} = Lay.ri*randn(Lay.nFilters, Lay.nJ_filt*Lay.nI_filt*Lay.nVisChannels);
        if isfield(Lay, 'initB')
            initB = Lay.initB;
        else
            initB = 0;
        end        
        net.hb{L} = initB*ones(Lay.nVisChannels, 1);
        
    elseif strcmp(Lay.type, 'convs')
    elseif strcmp(Lay.type, 'convjitter')
    elseif strcmp(Lay.type, 'convxyrs')
    elseif strcmp(Lay.type, 'imagemirror')
         
    elseif strcmp(Lay.type, 'convdata') ||  strcmp(Lay.type, 'fcdata')
        assert(false);
     end   
end
