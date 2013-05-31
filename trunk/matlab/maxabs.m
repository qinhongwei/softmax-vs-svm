function [err D] = maxabs(A, B)
D = abs(A-B);
err = max(max(D));
