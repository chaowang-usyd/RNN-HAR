function [llh,hd_end,hw_end,hm_end,VaR_end,QS_End] = VaR_HaR_llh(y,rv_all,theta_particles,alpha_level,act_fun)
% Calculate log-likelihood of VaR-HAR 
% INPUT
% y             : data
% RV             : realized measures
% and the model parameters
% returns data y, covariates x and model parameters
% OUTPUT
% llh           : log-likelihood

% alpha_d_0 = theta_particles(:,1);
% alpha_d_1 = theta_particles(:,2);
% alpha_d_2 = theta_particles(:,3);
% alpha_w_0 = theta_particles(:,4);
% alpha_w_1 = theta_particles(:,5);
% alpha_w_2 = theta_particles(:,6);
% alpha_m_0 = theta_particles(:,7);
% alpha_m_1 = theta_particles(:,8);
% alpha_m_2 = theta_particles(:,9);
beta0 = theta_particles(:,10);
beta1 = theta_particles(:,11);   
beta2 = theta_particles(:,12);
beta3 = theta_particles(:,13);
sigmaLH = theta_particles(:,14);

RVd = rv_all(:,1);
RVw = rv_all(:,2);
RVm = rv_all(:,3);

T = length(y);
h_d = (zeros(T,1));
h_w = (zeros(T,1));
h_m = (zeros(T,1));
Var_t = (zeros(T,1));
QS_t = (zeros(T,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
t = 1;
% h_d(t) =  activation(alpha_d_0 + alpha_d_1*RVd(t),act_fun);  
% h_w(t) =  activation(alpha_w_0 + alpha_w_1*RVw(t),act_fun);     
% h_m(t) =  activation(alpha_m_0 + alpha_m_1*RVm(t),act_fun);  

Var_t(t) = beta0;
QS_t(t)=((y(t)<Var_t(t))-alpha_level)*(Var_t(t)-y(t));


for t = 2:T
    Var_t(t) = beta0 + beta1*RVd(t-1) + beta2*RVw(t-1) + beta3*RVm(t-1);
    QS_t(t)=((y(t)<Var_t(t))-alpha_level)*(Var_t(t)-y(t));

    % h_d(t) =  activation(alpha_d_0 + alpha_d_1*RVd(t)+alpha_d_2*h_d(t-1),act_fun);  
    % h_w(t) =  activation(alpha_w_0 + alpha_w_1*RVw(t)+alpha_w_2*h_w(t-1),act_fun);     
    % h_m(t) =  activation(alpha_m_0 + alpha_m_1*RVm(t)+alpha_m_2*h_m(t-1),act_fun);  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
llh = sum(log(alpha_level*(1-alpha_level))-log(sigmaLH)-(y-Var_t).*(alpha_level-(y<Var_t))/sigmaLH);

hd_end = h_d(end);
hw_end = h_w(end);
hm_end = h_m(end);
VaR_end = Var_t(end);
QS_End = QS_t(end);

end
