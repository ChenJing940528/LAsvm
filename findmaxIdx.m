function [idx_I] = findmaxIdx(g,alpha,B)

ind_cond2 = find((alpha < B)==1);
ind_cond1 = find( (g == max(g)));
comb = intersect(ind_cond2 , ind_cond1);  % common elements
% if there is intersection.
if isempty(comb) == 1
   idx_I = [] ;
else
   idx_I = comb(randi(length(comb)));
end
end

