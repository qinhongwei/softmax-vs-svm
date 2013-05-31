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

    fc
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

function [ net ] = net_layers_deserialize( w, net_layers, nDataSources )

net.W = cell(1, length(net_layers)-nDataSources);
net.hb = cell(1, length(net_layers)-nDataSources);

for i = 1:nDataSources
    assert( strcmp(net_layers{i}.type, 'convdata') ... 
            || strcmp(net_layers{i}.type, 'fcdata') );
end

ind = 1;
for ll = (nDataSources+1):length(net_layers)
    
    Lay = net_layers{ll};
    L = ll-nDataSources;
    
     if strcmp(Lay.type, 'fc')
         
         ind2 = ind+Lay.nV*Lay.nH-1;
         net.W{L} = reshape(w(ind:ind2), Lay.nH, Lay.nV)';
         ind = ind2 + 1;
         
         ind2 = ind+Lay.nH-1;
         net.hb{L} = w(ind:ind2)';
         ind = ind2 + 1;
         
    elseif strcmp(Lay.type, 'convc')
        
        ind2 = ind+Lay.nVisChannels*Lay.nI_filt*Lay.nJ_filt*Lay.nFilters-1;
        net.W{L} = reshape(w(ind:ind2),  Lay.nFilters, Lay.nJ_filt*Lay.nI_filt*Lay.nVisChannels);
        ind = ind2 + 1;
    
        ind2 = ind+Lay.nFilters-1;
        net.hb{L} = w(ind:ind2);
        ind = ind2 + 1;
    elseif strcmp(Lay.type, 'deconvc')
        
        ind2 = ind+Lay.nVisChannels*Lay.nI_filt*Lay.nJ_filt*Lay.nFilters-1;
        net.W{L} = reshape(w(ind:ind2),  Lay.nFilters, Lay.nJ_filt*Lay.nI_filt*Lay.nVisChannels);
        ind = ind2 + 1;
    
        ind2 = ind+Lay.nVisChannels-1;
        net.hb{L} = w(ind:ind2);
        ind = ind2 + 1;
        
    elseif strcmp(Lay.type, 'convs')
    elseif strcmp(Lay.type, 'convrn')
    elseif strcmp(Lay.type, 'convjitter')
    elseif strcmp(Lay.type, 'convxyrs')
    elseif strcmp(Lay.type, 'imagemirror')
   
    elseif strcmp(Lay.type, 'convdata') ||  strcmp(Lay.type, 'fcdata')
        assert(false);     
    else
         assert(false);
    end
   
end


assert(ind == numel(w)+1);
