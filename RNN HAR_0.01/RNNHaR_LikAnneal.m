function Post = RNNHaR_LikAnneal(y,mdl)
% Implement the SMC likelihood annealing sampler for RNN HAR model
% y             : training data
% mdl           : data structure contains all neccesary settings
%
%
% @ Written by Minh-Ngoc Tran (minh-ngoc.tran@sydney.edu.au)
rv_all = mdl.rv_all;

% Training setting
T_anneal    = mdl.T_anneal;  % number of annealing steps
M           = mdl.M;         % number of particles
K1          = mdl.K1_lik;    % number of Markov moves in early levels
K2          = mdl.K2_lik;    % number of Markov moves in later levels, K2>=K1
prior       = mdl.prior;           % store prior setting
alpha_level = mdl.alpha_level; 
act_fun     = mdl.act_fun;

% Initialize particles for the first level by generating from the parameters priors
 alpha_d_0  = normrnd(prior.alpha_d0_mu,prior.alpha_d0_var,M,1);
 alpha_d_1  = normrnd(prior.alpha_d1_mu,prior.alpha_d1_var,M,1);
 alpha_d_2  = normrnd(prior.alpha_d2_mu,prior.alpha_d2_var,M,1);
 alpha_w_0  = normrnd(prior.alpha_w0_mu,prior.alpha_w0_var,M,1);
 alpha_w_1  = normrnd(prior.alpha_w1_mu,prior.alpha_w1_var,M,1);
 alpha_w_2  = normrnd(prior.alpha_w2_mu,prior.alpha_w2_var,M,1);
 alpha_m_0  = normrnd(prior.alpha_m0_mu,prior.alpha_m0_var,M,1);
 alpha_m_1  = normrnd(prior.alpha_m1_mu,prior.alpha_m1_var,M,1);
 alpha_m_2  = normrnd(prior.alpha_m2_mu,prior.alpha_m2_var,M,1);
 beta0 = normrnd(prior.beta0_mu,prior.beta0_var,M,1);
 beta1 = normrnd(prior.beta1_mu,prior.beta1_var,M,1);
 beta2 = normrnd(prior.beta2_mu,prior.beta2_var,M,1);
 beta3 = normrnd(prior.beta3_mu,prior.beta3_var,M,1);
 sigmaLH = 1./gamrnd(prior.sigmaLH_a, 1/prior.sigmaLH_b,M,1);

 theta_particles = [alpha_d_0, alpha_d_1, alpha_d_2, alpha_w_0, alpha_w_1, alpha_w_2, alpha_m_0, alpha_m_1, alpha_m_2, beta0, beta1, beta2, beta3, sigmaLH]; 

% Prepare for first annealing stage
psisq    = ((0:T_anneal)./T_anneal).^3;     % Design the array of annealing levels
log_llh  = 0;                               % for calculating marginal likelihood
n_params = size(theta_particles,2);        % number of model parameters, 15

% Calculate log likelihood for all particles in the first annealing level
llh_calc = zeros(M,1);  % log-likelihood calculated at each particle

for i = 1:M % each particle corresponds to a value of log-likelihood 
   [llh_calc(i)] = RNN_HaR_llh(y,rv_all,theta_particles(i,:),alpha_level,act_fun);   
end
t = 1; 
psisq_current = psisq(t); % current annealling level a_t

markov_idx = 1; % to count the times when Markov move step is executed
annealing_start = tic;

while t < T_anneal+1 
     t = t+1;
          
     %% Select the next annealing level a_t and then do reweighting %% 
     % Select a_t if ESS is less than the threshold. Then, reweight the particles       
     incw = (psisq(t) - psisq_current).*llh_calc; 
                                        
     max_incw = max(incw); % calculating the max value of the Maximum likelihoods, 
     weights = exp(incw - max_incw);      % for numerical stabability
     weights = weights./sum(weights);     % Calculate new normalized weights for current level,
     ESS = 1/sum(weights.^2);             % Estimate ESS for particles in the current level
     
     while ESS >= 0.8*M
        t = t + 1;
        % Run until ESS at a certain level < 80%. If reach the last level,
        % the last level will be the next annealing level.
        if (t == T_anneal+1)
            incw = (psisq(t)-psisq_current).*llh_calc;
            max_incw = max(incw);
            weights = exp(incw-max_incw);
            weights = weights./sum(weights); % setting the new normalized weights
            ESS = 1/sum(weights.^2);
            break % terminate the while loop while ESS >= 0.8*M
        else % If not reach the final level -> keep checking ESS 
            incw = (psisq(t)-psisq_current).*llh_calc;
            max_incw = max(incw);
            weights = exp(incw-max_incw);
            weights = weights./sum(weights);
            ESS = 1/sum(weights.^2);
        end
     end
     disp(['L Current annealing level: ',num2str(t)])     
     psisq_current = psisq(t);
     log_llh = log_llh + log(mean(weights)) + max_incw; % log marginal likelihood
    
     % calculate the covariance matrix used in the Random Walk MH proposal.
     % This is the empirical covariance matrix of the particles.
     est = sum(theta_particles.*(weights*ones(1,n_params))); % est = 1*15
     aux = theta_particles - ones(M,1)*est;% aux = 2000*15, substract est from each theta particle
     V = aux'*diag(weights)*aux;      % V = 15*15 double     
     C = chol(mdl.MV_scale/n_params*V,'lower'); % 2.38/n_params is a theoretically optimal scale

     %% Resampling the particles %%
     indx     = utils_rs_multinomial(weights'); 
     indx     = indx'; %2000*1
     theta_particles = theta_particles(indx,:); % selected theta according to the ladder values of utils function
     llh_calc = llh_calc(indx,:); % the log-likelihood values need to match the resampled particles 
                  
     %% Markov move step %%          
     accept       = zeros(M,1);   % to store acceptance rate in Markov move for each particle      
     log_prior    = RNN_HaR_logPriors(theta_particles,prior); %RNN_tGARCH_logPriors
     post         = log_prior+psisq_current*llh_calc;   % the denominator term in the Metropolis-Hastings ratio  , post = 2000*1 
     
     
     if psisq_current<0.7
         K = K1;
     else
         K = K2;
     end
     parfor i = 1:M  % Parallelize Markov move for the particles              
         iter = 1;
         while iter<=K                                        
             theta = theta_particles(i,:)'; % the particle to be moved
             % Using multivariate normal distribution as proposal
             theta_star = theta+C*normrnd(0,1,n_params,1); 
             theta_star = theta_star';
             % Convert parameters to original form
             sigmaLH_star = theta_star(14);               
             
            if (sigmaLH_star <= 0)        
                 acceptance_pro = 0; % acceptance probability is zero --> do nothing
             else                 
                 % Calculate log-posterior for proposal samples
                 log_prior_star = RNN_HaR_logPriors(theta_star,prior);
                 [lik_star]     = RNN_HaR_llh(y,rv_all,theta_star,alpha_level,act_fun);    
                 post_star      = log_prior_star + psisq_current*lik_star; % the numerator term in the Metropolis-Hastings ratio                  
                 acceptance_pro = exp(post_star-post(i));
                 acceptance_pro = min(1,acceptance_pro);                       
                 if (rand <= acceptance_pro) % if accept the new proposal sample
                     theta_particles(i,:) = theta_star;                     
                     post(i)              = post_star;
                     llh_calc(i)          = lik_star;
                     accept(i)            = accept(i) + 1;
                 end
             end             
             iter = iter + 1;
         end         
     end
     Post.accept_store(:,markov_idx) = accept/K; % store acceptance rate in Markov move
     disp(['L Markov move ',num2str(markov_idx),': L Avarage accept rate = ',num2str(mean(accept/K))])
     markov_idx = markov_idx + 1;
     
end
Post.cpu     = toc(annealing_start);
Post.alpha_d_0 = theta_particles(:,1);
Post.alpha_d_1 = theta_particles(:,2);
Post.alpha_d_2 = theta_particles(:,3);
Post.alpha_w_0 = theta_particles(:,4);
Post.alpha_w_1 = theta_particles(:,5);
Post.alpha_w_2 = theta_particles(:,6);
Post.alpha_m_0 = theta_particles(:,7);
Post.alpha_m_1 = theta_particles(:,8);
Post.alpha_m_2 = theta_particles(:,9);
Post.beta0    = theta_particles(:,10);
Post.beta1    = theta_particles(:,11);
Post.beta2     = theta_particles(:,12);
Post.beta3     = theta_particles(:,13);
Post.sigmaLH   = theta_particles(:,14);

Post.T_anneal = T_anneal;    
Post.M        = M;              
Post.K        = K;               
Post.log_llh  = log_llh;


disp(['L Marginal likelihood: ',num2str(log_llh)])


end