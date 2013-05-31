%{
Copyright (C) 2013 Yichuan Tang. contact: tang at cs.toronto.edu

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

%this function makes a normalized cross correlation type transformation
%Data is nsamples by ndimension matrix, CC is the final output
function [Data ] = ncc( Data, CC)

datamean = mean(Data,2);
Data = Data-datamean*ones(1,size(Data,2) );
datanorm = sqrt( sum(Data.*Data,2));

normeq0 = datanorm < 1e-8;
if any(normeq0)
    fprintf('*');
    datanorm(normeq0 ) = 1; %additional hack!!
end

Data = CC*Data./repmat(datanorm, 1, size(Data,2) );