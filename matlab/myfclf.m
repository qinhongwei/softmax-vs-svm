function [hdl] = myfclf(fignum, nR, nC, r, c)

hdl = sfigure(fignum);
clf;
%set(gcf, 'InitialMagnification', 'fit');
if nargin > 1
    subplot(nR, nC, r*nC+c+1);
end