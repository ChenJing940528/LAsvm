function [ s] = isKt_inS(x, x_Kt, S)

  s = 0;
  for i = 1:S
      s= s +1; 
     if  isequal (x(i,:), x_Kt) == 1   %x_kt is in set S         
         break    
     end
  end
  
  if  isequal (x(S,:), x_Kt) == 1  %x_Kt is in set S(end,:) 
      s = S;
  else
      s = S+3;
  end
  
end

