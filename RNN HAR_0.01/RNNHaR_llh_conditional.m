function [llh,Var_t_New,QS_t_New,hd_new,hw_new,hm_new] = RNNHaR_llh_conditional(y_new,y_cur,rv_new, rv_cur,hd_cur,hw_cur,hm_cur,VaR_cur,Qs_cur,theta_particles,alpha_level,act_fun)

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
beta1 = theta_particles(:,11);   
beta2 = theta_particles(:,12);
beta3 = theta_particles(:,13);
sigmaLH = theta_particles(:,14);

RVd_cur = rv_cur(:,1);
RVw_cur = rv_cur(:,2);
RVm_cur = rv_cur(:,3);

     hd_new =  activation(alpha_d_0 + alpha_d_1*RVd_cur +alpha_d_2.*hd_cur,act_fun);  
     hw_new =  activation(alpha_w_0 + alpha_w_1*RVw_cur +alpha_w_2.*hw_cur,act_fun);     
     hm_new =  activation(alpha_m_0 + alpha_m_1*RVm_cur +alpha_m_2.*hm_cur,act_fun);  

     Var_t_New = beta0 + beta1.*hd_new+ beta2.*hw_new + beta3.*hm_new;

     QS_t_New=((y_new<Var_t_New)-alpha_level).*(Var_t_New-y_new);

    llh = log(alpha_level*(1-alpha_level))-log(sigmaLH)-(y_new-Var_t_New).*(alpha_level-(y_new<Var_t_New))./sigmaLH; 

end