function [ Kout ] = Kernel(x_k,x,N)
% polynomial kernel k(x,y) = (xy'+1)^p, assume p =1
Kout = zeros(N,1);
for i =1:N
  Kout(i) = sum(x_k .*x(i,:))+1;
end

end

