function [ idx_out ]= findCandKt( cand, alpha, x_Train_rest, x, y, bias, M)
                 
 KA = zeros(M,1);
 
 for c =1:M
     KA(c,:) = sum(alpha'* Kernel( x_Train_rest(cand(c),:),x,length(y)))+bias;
     
    % KA(1,:) = sum(alpha(1:S)'* Kernel( x_Train_rest(idx_cand(1),:),x(1:S,:),length(y(1:S))))+bias
 end
 
 [~,idx_out] = min(KA);
 
end

