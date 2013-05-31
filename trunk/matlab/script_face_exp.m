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

clear;
global DataX;
global DataY;

DataX = [];
DataY = [];

addpath '../cuda_ut/bin/';
tic;
load data.mat;
toc;

bad_ind = find(sum(DataX.^2,2) < 1e-2);
DataX(bad_ind,:) = repmat(mean(DataX,1), length(bad_ind),1);

rand('seed',1234);
inds = randperm(size(DataX,1));
DataX = DataX(inds(1:128*14*16),:);
DataY = DataY(inds(1:128*14*16),:);

CC = 150
DataX = ncc( DataX, CC);

%%
hp =[];
hp.MaxIters = 50000;  %50k
hp.start_rate = 0.05;
hp.HalfLife = 20000; %10000
hp.momen = 0.9; %0.9
hp.noise1 = 0.0; %.0
hp.noise2 = 0.0; %.0
hp.nSamples = 128*2;
hp.momen_init = 5000;
hp.RM_MEAN_STD = 1;
hp.D = 48^2;
hp.net_layers = net_config_basic42();
hp.nJitterTrials = 64;

hp.randseeds = 12345;
hp.normalseeds = 1234;
hp.nSPLIT = 8; % 16

hp.the_splits =1:hp.nSPLIT;
%hp.the_splits = 1; %hack

gg;
[ cv_average, cv_models ] = ...
    fe_cv_48( ...
     hp.nSPLIT, hp.randseeds, hp.normalseeds, hp);
rg;

cv_average
mean(cv_average)

hp.CC = CC;
save 'model5.31.mat' cv_average cv_models hp;



                            
                            
