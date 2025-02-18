function results_collection

%%%%%%%%%%%%%% collect the parameter estiamtes
quantile_score= zeros(31,1);
parameter_estimates= zeros(31,13);

stock_name_list_temp    = {'AEX','AORD','BFX','BSESN','BVLG','BVSP','DJI','FCHI','FTMIB','FTSE',...
    'GDAXI','GSPTSE','HSI','IBEX','IXIC','KS11','KSE','MXX','N225','NSEI','OMXC20','OMXHPI','OMXSPI',...
    'OSEAX','RUT','SMSI','SPX','SSEC','SSMI','STI','STOXX50E'};

for jjj = 1:length(stock_name_list_temp)        
        str = 'Results_RNN_HAR_';
        stock_name = stock_name_list_temp{jjj};       
        str =append(str,stock_name);
        load (str)
        quantile_score(jjj)= Post_RNNHaR.DataAnneal.forecast.Quantile_Score
        para_temp= [mean(Post_RNNHaR.LikAnneal.beta0) mean(Post_RNNHaR.LikAnneal.beta1) mean(Post_RNNHaR.LikAnneal.beta2) mean(Post_RNNHaR.LikAnneal.beta3)... 
            mean(Post_RNNHaR.LikAnneal.alpha_d_0)  mean(Post_RNNHaR.LikAnneal.alpha_d_1)  mean(Post_RNNHaR.LikAnneal.alpha_d_2) ...
            mean(Post_RNNHaR.LikAnneal.alpha_w_0)  mean(Post_RNNHaR.LikAnneal.alpha_w_1)  mean(Post_RNNHaR.LikAnneal.alpha_w_2) ...
            mean(Post_RNNHaR.LikAnneal.alpha_m_0)  mean(Post_RNNHaR.LikAnneal.alpha_m_1)  mean(Post_RNNHaR.LikAnneal.alpha_m_2)];
        parameter_estimates(jjj,:)= para_temp;
end 


load Results_RNN_HAR_SPX.mat

subplot(1,4,1)
histogram(Post_RNNHaR.LikAnneal.beta0) 
xline(0); hold on; xline(mean(Post_RNNHaR.LikAnneal.beta0),'r',LineWidth=2)
title('\beta_0')
subplot(1,4,2)
histogram(Post_RNNHaR.LikAnneal.beta1)
xline(0);hold on; xline(mean(Post_RNNHaR.LikAnneal.beta1),'r',LineWidth=2)
title('\beta_d')
subplot(1,4,3)
histogram(Post_RNNHaR.LikAnneal.beta2) 
xline(0);hold on; xline(mean(Post_RNNHaR.LikAnneal.beta2),'r',LineWidth=2)
title('\beta_w')
subplot(1,4,4)
histogram(Post_RNNHaR.LikAnneal.beta3) 
title('\beta_m')
xline(0);hold on; xline(mean(Post_RNNHaR.LikAnneal.beta3),'r',LineWidth=2)



figure
subplot(3,3,1)
histogram(Post_RNNHaR.LikAnneal.alpha_d_0) 
xline(0); hold on; xline(mean(Post_RNNHaR.LikAnneal.alpha_d_0),'r',LineWidth=2)
title('\alpha_{0}^d')
subplot(3,3,2)
histogram(Post_RNNHaR.LikAnneal.alpha_d_1) 
xline(0); hold on; xline(mean(Post_RNNHaR.LikAnneal.alpha_d_1),'r',LineWidth=2)
title('\alpha_{1}^d')
subplot(3,3,3)
histogram(Post_RNNHaR.LikAnneal.alpha_d_2) 
xline(0); hold on; xline(mean(Post_RNNHaR.LikAnneal.alpha_d_2),'r',LineWidth=2)
title('\alpha_{2}^d')
subplot(3,3,4)
histogram(Post_RNNHaR.LikAnneal.alpha_w_0) 
xline(0); hold on; xline(mean(Post_RNNHaR.LikAnneal.alpha_w_0),'r',LineWidth=2)
title('\alpha_{0}^w')
subplot(3,3,5)
histogram(Post_RNNHaR.LikAnneal.alpha_w_1) 
xline(0); hold on; xline(mean(Post_RNNHaR.LikAnneal.alpha_w_1),'r',LineWidth=2)
title('\alpha_{1}^w')
subplot(3,3,6)
histogram(Post_RNNHaR.LikAnneal.alpha_w_2) 
xline(0); hold on; xline(mean(Post_RNNHaR.LikAnneal.alpha_w_2),'r',LineWidth=2)
title('\alpha_{2}^w')
subplot(3,3,7)
histogram(Post_RNNHaR.LikAnneal.alpha_m_0) 
xline(0); hold on; xline(mean(Post_RNNHaR.LikAnneal.alpha_m_0),'r',LineWidth=2)
title('\alpha_{0}^m')
subplot(3,3,8)
histogram(Post_RNNHaR.LikAnneal.alpha_m_1) 
xline(0); hold on; xline(mean(Post_RNNHaR.LikAnneal.alpha_m_1),'r',LineWidth=2)
title('\alpha_{1}^m')
subplot(3,3,9)
histogram(Post_RNNHaR.LikAnneal.alpha_m_2) 
xline(0); hold on; xline(mean(Post_RNNHaR.LikAnneal.alpha_m_2),'r',LineWidth=2)
title('\alpha_{2}^m')



posterior_estimate= mean(Post_RNNHaR.LikAnneal.alpha_m_2)
LB= quantile(Post_RNNHaR.LikAnneal.alpha_m_2,0.025)
UB= quantile(Post_RNNHaR.LikAnneal.alpha_m_2,0.975)
credible_interval= [posterior_estimate;LB;UB]'
