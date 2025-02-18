diary on;
diary('RNNHARSMC001.txt')

% This example uses SMC -likelihood annealing for samppling from RNNHaR model.
tic
clc;
clear;


%% prepare data

% ==============================================%
% %MNT: run all indexes
stock_name_list    = {'AEX','AORD','BFX','BSESN','BVLG','BVSP','DJI','FCHI','FTMIB','FTSE',...
    'GDAXI','GSPTSE','HSI','IBEX','IXIC','KS11','KSE','MXX','N225','NSEI','OMXC20','OMXHPI','OMXSPI',...
    'OSEAX','RUT','SMSI','SPX','SSEC','SSMI','STI','STOXX50E'};


for j = 1:length(stock_name_list)
    stock_name = stock_name_list{j};       
    disp('=========== stock name =====================')
    stock_name
    disp('==============================================')
    
      %% prepare data
    data       = load('Realizeddata2000To2022.mat');
    price      = data.(stock_name).close_price; 
    y_all_full      = log(price(2:end)./price(1:end-1)); % the first day isn't included
    y_all_full      =  100*(y_all_full-mean(y_all_full)); % returns (from the 2nd day)
    rv_all_Original     = 10^4*data.(stock_name).rv5(2:end); % realized volatility; rv_all(t) realizes the volatility of y_all(t)
    rv_all_Original = sum(y_all_full.^2)./sum(rv_all_Original)*rv_all_Original; % rescale (using training data) to reflect overnight variation    
    
    % creating daily, weekly, monthly series.
    % daily rv
    rv_d = rv_all_Original;
   % weekly rv
   % Initialize an array for the weekly rv.
    % Calculate the moving average using a for loop
    rv_w = nan(size(rv_all_Original, 1), 1);
    rv_m = nan(size(rv_all_Original, 1), 1);

for i = 1:numel(rv_all_Original) 
    % Check if there are enough elements to calculate the average
    if i >= 5
        % Calculate the mean for the window
        rv_w(i) = mean(rv_all_Original(i-5+1:i));
    else
        % If not enough elements, set the value to NaN
        rv_w(i) = NaN;
    end
end

% Initialize an array for the monthly rv.
% Calculate the moving average using a for loop
for i = 1:numel(rv_all_Original) 
    % Check if there are enough elements to calculate the average
    if i >= 22
        % Calculate the mean for the window
        rv_m(i) = mean(rv_all_Original(i-22+1:i));
    else
        % If not enough elements, set the value to NaN
        rv_m(i) = NaN;
    end
end

RV_matrix_Full = [rv_d rv_w rv_m];

    RV = RV_matrix_Full(22:end,:);
    y_all = y_all_full(22:end);

    if length(y_all) > 3000
        y_all      = y_all(end-2999:end); % use the last 3000 days        
        rv_all     = RV(end-2999:end,:); 
        T          = 2000; % training size
    else
        T = round(length(y_all)*2/3);
        rv_all     = RV;
    end        
    y          = y_all(1:T); 
    rv         = rv_all(1:T,:); 
    mdl.rv_all = rv_all;
    clear data

% Training setting
mdl.T_anneal = 10000;    % Number of pre-specified annealing steps
mdl.M        = 2000;     % Number of particles in each annealing stage
mdl.K1_lik   = 10;       % Number of Markov moves 
mdl.K2_lik   = 20;       % Number of Markov moves 
mdl.T        = T;     % size of the training time series 
y            = y_all(1:mdl.T);  % training data 2000*1
mdl.MV_scale = 0.1;% works good 40%-50% acceptance rate
mdl.alpha_level = 0.01;   
mdl.act_fun = 'Tanh';

% Priors 
mdl.prior.alpha_d0_mu = 0;  mdl.prior.alpha_d0_var = 0.01;    % Normal distribution
mdl.prior.alpha_d1_mu = 0;  mdl.prior.alpha_d1_var = 0.01;    % Normal distribution
mdl.prior.alpha_d2_mu = 0;  mdl.prior.alpha_d2_var = 0.01;    % Normal distribution
mdl.prior.alpha_w0_mu = 0;  mdl.prior.alpha_w0_var = 0.01;    % Normal distribution
mdl.prior.alpha_w1_mu = 0;  mdl.prior.alpha_w1_var = 0.01;    % Normal distribution
mdl.prior.alpha_w2_mu = 0;  mdl.prior.alpha_w2_var = 0.01;    % Normal distribution
mdl.prior.alpha_m0_mu = 0;  mdl.prior.alpha_m0_var = 0.01;    % Normal distribution
mdl.prior.alpha_m1_mu = 0;  mdl.prior.alpha_m1_var = 0.01;    % Normal distribution
mdl.prior.alpha_m2_mu = 0;  mdl.prior.alpha_m2_var = 0.01;    % Normal distribution

mdl.prior.beta0_mu = 0;  mdl.prior.beta0_var = 1;    % Normal distribution
mdl.prior.beta1_mu = 0;  mdl.prior.beta1_var = 1;    % Normal distribution, 
mdl.prior.beta2_mu = 0;  mdl.prior.beta2_var = 1;    % Normal distribution
mdl.prior.beta3_mu = 0;  mdl.prior.beta3_var = 1;    % Normal distribution
mdl.prior.sigmaLH_a = 1;  mdl.prior.sigmaLH_b = 1./1; % Inverse gamma distribution

% Run Likelihood annealing for in-sample data
Post_RNNHaR.mdl = mdl;
Post_RNNHaR.LikAnneal = RNNHaR_LikAnneal(y,mdl); 

% Forecast with data annealing
mdl.lik_anneal          = Post_RNNHaR.LikAnneal;
mdl.MV_scale            = 0.1;%0.1;%0.2 
mdl.K_data              = 20; % 15 works good 40%-50% acceptance rate

Post_RNNHaR.DataAnneal = RNNHaR_DataAnneal(y_all,mdl); %RECH_DataAnneal

    str = 'Results_RNN_HAR_';
    str = append(str,stock_name);    
    save(str)
    

end
diary off;












