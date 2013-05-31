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

    net{LL}.W
    net{LL}.hb
    
TESTED:
TODO:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function [w] = net_layers_serialize( net, net_layers, nDataSources )

assert( length(net.W) == length(net_layers)-nDataSources);
assert( length(net.hb) == length(net_layers)-nDataSources);

for i = 1:nDataSources
    assert( strcmp(net_layers{i}.type, 'convdata') ... 
            || strcmp(net_layers{i}.type, 'fcdata') );
end
        
dd = 0;
for ll = (nDataSources+1):length(net_layers)
    Lay = net_layers{ll};
    L = ll-nDataSources;
    
    if strcmp(Lay.type, 'fc') 
        assert(all( size(net.W{L}) == [Lay.nV Lay.nH]));
        dd = dd+numel(net.W{L});
        assert(all( size(net.hb{L}) == [1 Lay.nH]));
        dd = dd+numel(net.hb{L});
        
        
    elseif strcmp(Lay.type, 'convc')
                
        assert(size(net.W{L},1) ==  Lay.nFilters);
        assert(size(net.W{L},2) == Lay.nJ_filt*Lay.nI_filt*Lay.nVisChannels);
        assert(size(net.hb{L},1) == Lay.nFilters && size(net.hb{L},2) == 1);
        
        dd = dd+numel(net.W{L});
        dd = dd+numel(net.hb{L});
     
    elseif strcmp(Lay.type, 'deconvc')
                
        assert(size(net.W{L},1) ==  Lay.nFilters);
        assert(size(net.W{L},2) == Lay.nJ_filt*Lay.nI_filt*Lay.nVisChannels);
        assert(size(net.hb{L},1) == Lay.nVisChannels && size(net.hb{L},2) == 1);
        
        dd = dd+numel(net.W{L});
        dd = dd+numel(net.hb{L});
                        
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


w = zeros(dd,1);
ind = 1;

for ll = (nDataSources+1):length(net_layers)
    Lay = net_layers{ll};
    L = ll-nDataSources;
    
    if strcmp(Lay.type, 'fc')
        
        ind2 = ind+numel(net.W{L})-1;
        w(ind:ind2) = sc( net.W{L}')';
        ind = ind2+1;
        
        ind2 = ind+numel(net.hb{L})-1;
        w(ind:ind2) = sc( net.hb{L})';
        ind = ind2+1;       
              
    elseif strcmp(Lay.type, 'convc') || strcmp(Lay.type, 'deconvc')
                
        ind2 = ind+numel(net.W{L})-1;
        w(ind:ind2) = sc( net.W{L})';
        ind = ind2+1;
        
        ind2 = ind+numel(net.hb{L})-1;
        w(ind:ind2) = sc(net.hb{L})';
        ind = ind2+1;
        
    elseif strcmp(Lay.type, 'convs')
    elseif strcmp(Lay.type, 'convjitter')
    elseif strcmp(Lay.type, 'convxyrs')
    elseif strcmp(Lay.type, 'imagemirror')
   
    elseif strcmp(Lay.type, 'convdata') ||  strcmp(Lay.type, 'fcdata')
        assert(false);
    end
end

assert(ind == dd+1);
