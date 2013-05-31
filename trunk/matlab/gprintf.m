
%usage:   
%   instead of fprintf, use gprintf
function [] = gprintf(msg, varargin )

txtmsg = sprintf(msg, varargin{1:end} );
figure(999); clf;
axis off;
set(gcf, 'Color', 'white');
text( -0.1, 0.5, txtmsg, 'EdgeColor', 'blue', 'FontSize', 30);