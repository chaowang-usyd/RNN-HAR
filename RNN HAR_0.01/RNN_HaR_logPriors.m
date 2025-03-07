function log_prior = RNN_HaR_logPriors(theta_particles,prior)

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


log_prior = log(normpdf(alpha_d_0,prior.alpha_d0_mu,sqrt(prior.alpha_d0_var)))...
    +log(normpdf(alpha_d_1,prior.alpha_d1_mu,sqrt(prior.alpha_d1_var)))...
    +log(normpdf(alpha_d_2,prior.alpha_d2_mu,sqrt(prior.alpha_d2_var)))...
    +log(normpdf(alpha_w_0,prior.alpha_w0_mu,sqrt(prior.alpha_w0_var)))...
    +log(normpdf(alpha_w_1,prior.alpha_w1_mu,sqrt(prior.alpha_w1_var)))...
    +log(normpdf(alpha_w_2,prior.alpha_w2_mu,sqrt(prior.alpha_w2_var)))...
    +log(normpdf(alpha_m_0,prior.alpha_m0_mu,sqrt(prior.alpha_m0_var)))...
    +log(normpdf(alpha_m_1,prior.alpha_m1_mu,sqrt(prior.alpha_m1_var)))...
    +log(normpdf(alpha_m_2,prior.alpha_m2_mu,sqrt(prior.alpha_m2_var)))...
    + log(normpdf(beta0,prior.beta0_mu,sqrt(prior.beta0_var))) ...
    + log(normpdf(beta1,prior.beta1_mu,sqrt(prior.beta1_var)))...
    + log(normpdf(beta2,prior.beta2_mu,sqrt(prior.beta2_var)))...
    + log(normpdf(beta3,prior.beta3_mu,sqrt(prior.beta3_var)))...
    + log_invgampdf(sigmaLH,prior.sigmaLH_a,prior.sigmaLH_b);
  
end
