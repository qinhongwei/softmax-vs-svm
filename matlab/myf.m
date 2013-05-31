function [hdl] = myf(fignum, nR, nC, r, c)

hdl = sfigure(fignum);

if nargin > 1
    subplot(nR, nC, r*nC+c+1);
end