data=readtable('time14102102.csv', 'Delimiter', ',');
data2=readtable('time141021.csv', 'Delimiter', ',');

%% Initialization
x_o = [data2.C14,data.C15];
y_o = data2.click;  %let y = 0  ==>-1
y_o(y_o == 0) = -1; 
% Normalize Sample Points  ?????
x_o = (x_o-mean(x_o))./std(x_o);
  % y = (y-mean(y))./std(y);
  
N_1 = length(y_o);  
%% Validiatation set---> rand select 1/11
idx_all= randperm(N_1)';
x_V = x_o( idx_all(1:round(1/11*N_1)), :);
y_V = y_o( idx_all(1:round(1/11*N_1)));
%% Train_set --> the rest 10/11  ---> 
x_Train = x_o( idx_all(round(1/11*N_1)+1:end), :);
y_Train = y_o( idx_all(round(1/11*N_1)+1:end) );


%rand select S =1000 examples from Train set into set S
S = 100;
idx_Train = randperm(length(y_Train))'; 

%% for speed
SS = 4000000;
x = zeros(SS,size(x_Train,2));    
y = zeros(SS,1);

x(1:S,:) = x_Train( idx_Train(1:S), :);
y(1:S,:) = y_Train( idx_Train(1:S) );

%% divide the rest of examples from Train into N_train/M 'ind_cand()' groups
x_Train_rest = x_Train( idx_Train(S+1:end), :);
y_Train_rest = y_Train( idx_Train(S+1:end) );

idx_Train_rest = randperm(length(y_Train_rest))';
M = 80;  % length(idx_Train_rest)/M %%%% where M needs to change for each cell
         % and pick up one of 50 which 'argmin_kt | f(x_kt) |' ;
         % divided the rest of Train data into N_train_rest / M groups. 
mod_M = mod(length(idx_Train_rest) , M);        
Groups = reshape( idx_Train_rest(1:end-mod_M) , M, (length(idx_Train_rest)-mod_M)/M );
% --> each column denotes one  group;  % size_Train_Group = size(Groups);% 80   2345  

 
% initialization
alpha = zeros(SS,1);
g = zeros(SS,1);

for k =1:S
   g(k) = y(k) - sum(alpha(1:S).* Kernel(x(k,:),x(1:S,:),S));
end


%% Online iterations--
passes = 0;
% max_passes = % ??
tol = 10^(-5);  %% cross validation ???
C = 0.5;     %% ??/
bias = 0;

% while (passes < max_passes)

  %% 2) pick an example x_Kt (from M random samples)
 % while % S < 1050
  for Runs = 1 : 100 % should be first 10 of 2345 groups 
  idx_cand = Groups(:,Runs);  % there are 2345 cols/groups
  % find one in 50 which 'argmin_kt | f(x_kt) |' 
  idx_Kt = findCandKt(idx_cand, alpha(1:S), x_Train_rest, x(1:S,:), y(1:S), bias, M);   
  
  %% PROCESS(Kt)
  % 2)bail out if x_Kt belongs to S
  if isempty ( find(x(1:S,:) == x(idx_Kt,:), 1) ) == 0 
      continue;   %go to next Runs to find new Kt
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
      
 %2) bail out if (i,j) is not a tuta-violating pair; omit (alpha(I) >= B(I)) || (alpha(J) <= A(J)) || 
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
  
 end % end for Runs
% end % end for while

% end while  % skips to next outerIdx
