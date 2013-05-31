% Seralize in Column major fashion: same as (:), but we can use
% e.g. sc( im(1:10,1:10)' )

function [ rowvec ] = sc( image )
rowvec = image(:)';