function Post = RNNHaR_DataAnneal(y_all,mdl)% r_all = 2626*1, md1: rv_all and x_all = 2626*1
% Implement SMC- data annealing for for RECH (RNN-GARCH)
%   y_all           : full dataset including both training and testing data          
%   mdl             : includes all necessary settings, including posterior
%                     approximation from SMC likelihhood annealing with
%                     y_train data
%
% @ Written by Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)


%% Training
M           = mdl.M;            % Number of particles
K          = mdl.K_data;        % number of Markov moves in early levels
prior       = mdl.prior;         % prior setting
T           = mdl.T;             % training data size,
alpha_level = mdl.alpha_level; 
act_fun     = mdl.act_fun;

y  = y_all(1:T);   
y_test = y_all(T+1:end);

rv_all  = mdl.rv_all;
rv      = rv_all(1:T,:);
rv_test = rv_all(T+1:end,:); 

T_test      = length(y_test); 

% Forecast score metrics

score.qs      = 0;  
score.alpha   = alpha_level; 


% Get equally-weighted particles from SMC lik annealing as the initial particles 
alpha_d_0 = mdl.lik_anneal.alpha_d_0;
alpha_d_1 = mdl.lik_anneal.alpha_d_1;
alpha_d_2 = mdl.lik_anneal.alpha_d_2;
alpha_w_0 = mdl.lik_anneal.alpha_w_0;
alpha_w_1 = mdl.lik_anneal.alpha_w_1;
alpha_w_2 = mdl.lik_anneal.alpha_w_2;
alpha_m_0 = mdl.lik_anneal.alpha_m_0;
alpha_m_1 = mdl.lik_anneal.alpha_m_1;
alpha_m_2 = mdl.lik_anneal.alpha_m_2;
beta0 = mdl.lik_anneal.beta0;
beta1 = mdl.lik_anneal.beta1;
beta2 = mdl.lik_anneal.beta2;
beta3 = mdl.lik_anneal.beta3;
sigmaLH = mdl.lik_anneal.sigmaLH;

theta_particles = [alpha_d_0, alpha_d_1, alpha_d_2, alpha_w_0, alpha_w_1, alpha_w_2, alpha_m_0, alpha_m_1, alpha_m_2, beta0, beta1, beta2, beta3, sigmaLH]; 
Weights   = ones(M,1)./M;    % Initialize equal weights for articles in the first level
n_params  = size(theta_particles,2); % number of model parameters

% Run GARCH on training data to get initialization on test data
llh_calc = zeros(M,1);   % log-likelihood 
hd_cur = zeros(M,1);     % Store condiWeightstional variance of the current distribution
hw_cur = zeros(M,1);   
hm_cur = zeros(M,1);   
VaR_cur = zeros(M,1);   
QS_cur = zeros(M,1); 

for i = 1:M
    [llh_calc(i),hd_cur(i),hw_cur(i),hm_cur(i),VaR_cur(i),QS_cur(i)] = RNN_HaR_llh(y,rv,theta_particles(i,:),alpha_level,act_fun);        
end 

markov_idx = 0;
annealing_start = tic;

for t = 0:T_test-1 
    %% 1-step-ahead VaR forecast %%
    if t>0 % get current data point y_cur
        y_cur = y_test(t);  % t>0 then get normal y test value
    else
        y_cur = y(T); % if t = 0 then the last value of the insample
    end
    
    if t>0 % get current data point RV_cur
        rv_cur = rv_test(t,:);  
    else
        rv_cur = rv(T,:);
    end
%
%% Calculating score 1 using by default weights following Nick's code   
   [Var_forecast] = RNNHaR_one_step_forecast(y_cur,rv_cur,hd_cur,hw_cur,hm_cur,VaR_cur,QS_cur,theta_particles,alpha_level,act_fun); 
   
    VaR_next(t+1) = Weights'*Var_forecast;  

    score = one_step_forecast_score(VaR_next(t+1),y_test(t+1),score); 
%     score  



%% Re-weighting %%
    % Calculate log conditional likelihood p(y_t+1|y_1:t,theta)
    if t==0
        [llh_condtional,VaR_cur,Qs_cur,hd_cur,hw_cur,hm_cur] = RNNHaR_llh_conditional(y_test(t+1),y(T),rv_test(t+1,:),rv(T,:),hd_cur,hw_cur,hm_cur,VaR_cur,QS_cur,theta_particles,alpha_level,act_fun);
    else  % RNN_tGARCH_llh_conditional
        [llh_condtional,VaR_cur,Qs_cur,hd_cur,hw_cur,hm_cur] = RNNHaR_llh_conditional(y_test(t+1),y_test(t),rv_test(t+1,:),rv_test(t,:),hd_cur,hw_cur,hm_cur,VaR_cur,QS_cur,theta_particles,alpha_level,act_fun);
    end                                                         
    % Update the log likelihood p(y_{1:t+1})

    llh_calc = llh_calc + llh_condtional;
    % Reweighting the particles  
    incw = log(Weights) + llh_condtional;
    max_incw = max(incw);
    weight   = exp(incw - max_incw);    % for numerical stabability
    Weights  = weight./sum(weight);     % Calculate weights for current level
    ESS      = 1/sum(Weights.^2);       % Estimate ESS for particles in the current level     
    
   
    %% If the current particles are not good, run resampling and Markov move
    if (ESS < 0.80*M)
        disp(['D Current t: ',num2str(t)])     
        % calculate the covariance matrix to be used in the Random Walk MH
        % proposal. It is better to estimate this matrix BEFORE resampling
        est = sum(theta_particles.*(Weights*ones(1,n_params)));
        aux = theta_particles - ones(M,1)*est;
        V = aux'*diag(Weights)*aux;    
        C = chol(mdl.MV_scale/n_params*V,'lower'); % 2.38/n_params is a theoretically optimal scale, used in Markov move
 
        % Resampling for particles at the current annealing level
        indx            = utils_rs_multinomial(Weights');
        indx            = indx';
        theta_particles = theta_particles(indx,:);
        llh_calc        = llh_calc(indx);
        hd_cur      = hd_cur(indx);
        hw_cur      = hw_cur(indx);
        hm_cur      = hm_cur(indx);  
        VaR_cur    = VaR_cur(indx);
        Qs_cur       = Qs_cur(indx);
        Weights = ones(M,1)./M; 
      
        % Running Markov move (MH) for each paticles
        markov_idx = markov_idx + 1;
        accept = zeros(M,1);
        log_prior    = RNN_HaR_logPriors(theta_particles,prior);        
        post         = log_prior+llh_calc;
     
        parfor i = 1:M
            iter = 1;
            while iter<=K
             theta = theta_particles(i,:);
                 % Using multivariate normal distribution as proposal function
             theta_star = theta'+C*normrnd(0,1,n_params,1);
             theta_star = theta_star';
                 % Convert parameters to original form
             sigmaLH_star = theta_star(14);  
             
               if (sigmaLH_star <= 0)%          
                 acceptance_pro = 0; % acceptance probability is zero --> do nothing
               else                 
                     % Calculate log-posterior for proposal samples
                     log_prior_star = RNN_HaR_logPriors(theta_star,prior);
                     [lik_star,hd_new_star,hw_new_star,hm_new_star,VaR_new_star,QS_new_star] = RNN_HaR_llh(y_all(1:T+t+1),rv_all(1:T+t+1,:),theta_star,alpha_level,act_fun);                         
                     post_star      = log_prior_star + lik_star; % the numerator term in the Metropolis-Hastings ratio 
                     acceptance_pro = exp(post_star-post(i));
                     acceptance_pro = min(1,acceptance_pro);                       
                     if (rand <= acceptance_pro) % if accept the new proposal sample
                         theta_particles(i,:) = theta_star;                     
                         post(i)              = post_star;
                         llh_calc(i)          = lik_star;
                         hd_cur(i)          = hd_new_star;
                         hw_cur(i)          = hw_new_star;
                         hm_cur(i)          = hm_new_star;
                         VaR_cur(i)         = VaR_new_star;
                         QS_cur(i)          = QS_new_star;
                         accept(i)          = accept(i) + 1;                         
                     end
               end  
                 iter = iter + 1;
            end             
        end
         Post.accept_store(:,markov_idx) = accept/K;
         disp(['D Markov move ',num2str(markov_idx),': D Avarage accept rate = ',num2str(mean(accept/K))])
       
    end

end

Post.cpu     = toc(annealing_start);
Post.alpha_d_0   = theta_particles(:,1);
Post.alpha_d_1   = theta_particles(:,2);
Post.alpha_d_2    = theta_particles(:,3);
Post.alpha_w_0    = theta_particles(:,4);
Post.alpha_w_1    = theta_particles(:,5);
Post.alpha_w_2    = theta_particles(:,6);
Post.alpha_m_0    = theta_particles(:,7);
Post.alpha_m_1    = theta_particles(:,8);
Post.alpha_m_2   = theta_particles(:,9);
Post.beta0    = theta_particles(:,10);
Post.beta1    = theta_particles(:,11);
Post.beta2    = theta_particles(:,12);
Post.beta3    = theta_particles(:,13);
Post.sigmaLH  = theta_particles(:,14);
Post.M        = M;              
Post.K        = K;               
Post.Weights  = Weights;

Post.VaR_next = VaR_next;
Quantile_Score = score.qs/T_test;  
score.Quantile_Score = Quantile_Score;
Post.score    = score;  
Model = {'HAR-RNN'};
results = table(Model,Quantile_Score);
Post.forecast = results;
disp(results);

end







