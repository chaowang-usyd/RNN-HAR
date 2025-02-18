function [Var_forecast] = RNNHaR_one_step_forecast(y_cur,rv_cur,hd_cur,hw_cur,hm_cur,VaR_cur,QS_cur,theta_particles,alpha_level,act_fun)

alpha_d_0 = theta_particles(:,1);
alpha_d_1 = theta_particles(:,2);
alpha_d_2 = theta_particles(:,3);
alpha_w_0 = theta_particles(:,4);
alpha_w_1 = theta_particles(:,5);
alpha_w_2 = theta_particles(:,6);
alpha_m_0 = theta_particles(:,7);
alpha_m_1 = theta_particles(:,8);
alpha_m_2 = theta_particles(:,9);
beta0 = theta_particles(:,10);
%beta1 = 1./(1+exp(-theta_particles(:,11)));
beta1 = theta_particles(:,11);   
beta2 = theta_particles(:,12);
beta3 = theta_particles(:,13);
sigmaLH = theta_particles(:,14);

RVd_cur = rv_cur(:,1);
RVw_cur = rv_cur(:,2);
RVm_cur = rv_cur(:,3);

     hd_forecast =  activation(alpha_d_0 + alpha_d_1 .* RVd_cur + alpha_d_2 .* hd_cur,act_fun);  
     hw_forecast =  activation(alpha_w_0 + alpha_w_1 .* RVw_cur + alpha_w_2 .* hw_cur,act_fun);     
     hm_forecast =  activation(alpha_m_0 + alpha_m_1 .* RVm_cur + alpha_m_2 .* hm_cur,act_fun);  

     Var_forecast= beta0 + beta1.*hd_forecast + beta2.*hw_forecast + beta3.*hm_forecast;

end