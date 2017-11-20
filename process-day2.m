%% for day 2 
data2 = load ('day2.mat');
data2 = data2.data_impctr;
%% Initialization
 %    site_join_imp; device_type; C14; banner_pos; ctr_C15;....
x_o = [data2.imp_site_out_num, data2.device_type, data2.C14, data2.banner_pos, data2.ctr_C15];
y_o = data2.click;  %let y = 0  ==>-1
y_o(y_o == 0) = -1; 
% Normalize continuous Sample Points [ data.imp_site_out_num--col 1 ;  data.ctr_C15 --col 5]
x_o(:,[1,5]) = (x_o(:,[1,5])-mean(x_o(:,[1,5])))./std(x_o(:,[1,5]));
% x_o.Properties.VariableNames = {'imp_site_out_num' 'devicetype' 'C14' 'bannerpos' 'ctrC15'};

d = size(x_o,2);  
N_1 = length(y_o);  


%% Test set---> rand select 1/11
idx_all= randperm(N_1)';
x_Test = x_o( idx_all(1:round(1/11*N_1)), :);
y_Test = y_o( idx_all(1:round(1/11*N_1)));
Test_set = length(y_Test);
y_pred = zeros(Test_set,1);
for k =1:Test_set
    y_pred(k) = sum(alpha(1:S)'* Kernel(x_Test(k,:),x(1:S,:),S)) + bias;
end

y_pred(y_pred > 0) = 1;
y_pred(y_pred < 0) = -1;
% correct rate
(length(find(y_Test - y_pred == 0))) / Test_set    % 0.8404



%% Train the rest 
%% Train_set --> the rest 9/11  
x_Train_rest = x_o( idx_all(round(1/11*N_1)+1:round(10/11*N_1)), :);
y_Train_rest = y_o( idx_all(round(1/11*N_1)+1:round(10/11*N_1)) );
%% Validation set ---> the rest 1/11
x_V_set = x_o( idx_all(round(10/11*N_1)+1:end), :);
y_V_set = y_o( idx_all(round(10/11*N_1)+1:end) );
idx_Train_rest = randperm(length(y_Train_rest))';
M = 80;  
mod_M = mod(length(idx_Train_rest) , M);        
Groups2 = reshape( idx_Train_rest(1:end-mod_M) , M, (length(idx_Train_rest)-mod_M)/M );
%% 2) pick an example x_Kt (from M random samples)
 % while % S < 1050
 tic
  for Runs = 1 : size(Groups2,2) % should be first 10 of 2345 groups 
 
  idx_cand = Groups(:,Runs);  % there are 2345 cols/groups
  % find one in M which 'argmin_kt | f(x_kt) |' 
  idx_Kt = findCandKt(idx_cand, alpha(1:S), x_Train_rest, x(1:S,:), y(1:S), bias, M);   
  
  %% PROCESS(Kt)
  % 2)bail out if x_Kt belongs to S
  if isKt_inS(x, x_Train_rest(idx_Kt,:), S) < S+2
      continue
  else
      x(S+1,:) = x_Train_rest(idx_Kt,:);   
      y(S+1) = y_Train_rest(idx_Kt,:);
      g(S+1) = y(S+1) - sum(alpha(1:S).* Kernel(x(S+1,:),x(1:S,:),S));
      alpha(S+1) = 0;    % alpha_Kt =0;
  end
 
  %3) compute A and B
  A= min(0,C*y(1:S+1));
  B= max(0,C*y(1:S+1));
  if y(S+1) ==1
      I = S+1;
      J = findminIdx(g(1:S+1),alpha(1:S+1),A);
  else
      J = S+1;
      I = findmaxIdx(g(1:S+1),alpha(1:S+1),B);
  end
  % bail out if (I, J) is [] or bail out !! 
  if (isempty(I) == 1) ||( isempty(J) == 1 )
      x(S+1,:) = [];
      y(S+1) =[];
      g(S+1) =[];
      alpha(S+1) = [];
      continue;
  end
      
  %4) bail out if (i,j) is not a tau-violating pair % omit (alpha(I) >= B(I)) || (alpha(J) <= A(J)) 
  if (g(I)- g(J)) <= tol
      x(S+1,:) = [];
      y(S+1) =[];
      g(S+1) =[];
      alpha(S+1) = [];
      continue;  % skips to next outerIdx   
  end
  
  %5) perform a direction search
  eta = Kernel(x(I,:),x(I,:),1) + Kernel(x(J,:),x(J,:),1) - 2* Kernel(x(I,:),x(J,:),1);
  lambda = min( [(g(I)-g(J))./eta , B(I)-alpha(I) , alpha(J)-A(J)] );
  alpha(I) = alpha(I) +lambda;
  alpha(J) = alpha(J) -lambda;
  g(1:S+1) = g(1:S+1) - lambda *( Kernel(x(I,:),x(1:S+1, :),S+1) - Kernel(x(J,:),x(1:S+1, :), S+1) );
  
%% REorder ????
%% REPROCESS
 %1)  
  I = findmaxIdx(g(1:S+1),alpha(1:S+1),B);
  J = findminIdx(g(1:S+1),alpha(1:S+1),A);
  % bail out if (I, J) is [] or bail out !! 
  if (isempty(I) == 1) ||( isempty(J) == 1 )
      continue;
  end
      
 %2) bail out if (i,j) is not a tao-violating pair; omit (alpha(I) >= B(I)) || (alpha(J) <= A(J)) || 
  if (g(I)- g(J)) <= tol
       continue;
  end
 %3) perform a direction search
  eta = Kernel(x(I,:),x(I,:),1) + Kernel(x(J,:),x(J,:),1) - 2* Kernel(x(I,:),x(J,:),1);
  lambda = min( [(g(I)-g(J))./eta , B(I)-alpha(I) , alpha(J)-A(J)] );
  alpha(I) = alpha(I) +lambda;
  alpha(J) = alpha(J) -lambda;
  g(1:S+1) = g(1:S+1) - lambda *( Kernel(x(I,:),x(1:S+1,:),S+1) - Kernel(x(J,:),x(1:S+1,:),S+1) );
  
 %4) remove blatant non suppot vectors
  I = findmaxIdx(g(1:S+1),alpha(1:S+1),B);
  J = findminIdx(g(1:S+1),alpha(1:S+1),A);
  % for all s such that alpha_s = 0
  idx_remove = find((alpha(1:S+1) == 0));
  for s = 1:idx_remove
      if (y(s) == -1) && (g(s)>g(I))
          x(s,:) = [];
          y(s) =[];
          g(s) =[];
          alpha(s) = [];
      elseif (y(s) == 1) && (g(s) <= g(J))
          x(s,:) = [];
          y(s) =[];
          g(s) =[];
          alpha(s) = [];
      end    
  end
 
  % 5) bias term
  bias = (g(I) + g(J))/2;
  sigma = g(I) - g(J);
  
  S = length(find(y~=0));   % the number of examples in set S 
  t = t+1;
  
  msg = sprintf('Run number %d; Bias is %d; sigma is %d; t is %d', Runs, bias, sigma, t);
  disp(msg)   
  
 end % end for Runs
 toc
% end % end for while

% end while  % skips to next outerIdx