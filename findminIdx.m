function [idx_J] = findminIdx(g,alpha,A)

ind_cond2 = find( (alpha > A)==1);
ind_cond1 = find( (g == min(g)));
comb = intersect(ind_cond2 , ind_cond1);  % common elements
% if there is intersection.
if isempty(comb) == 1
   idx_J = [] ;
else
   idx_J = comb(randi(length(comb)));
end

end

