
stock_name_list    = {'AEX','AORD','BFX','BSESN','BVLG','BVSP','DJI','FCHI','FTMIB','FTSE',...
    'GDAXI','GSPTSE','HSI','IBEX','IXIC','KS11','KSE','MXX','N225','NSEI','OMXC20','OMXHPI','OMXSPI',...
    'OSEAX','RUT','SMSI','SPX','SSEC','SSMI','STI','STOXX50E'};

% stock_name_list    = {'AEX'};
% j = 1;
 for j = 1:length(stock_name_list)
    stock_name = stock_name_list{j};       
    disp('=========== stock name =====================')
    stock_name
    disp('==============================================')
    
      %% prepare data
    file_name = ['Results_RNN_HAR_' stock_name '.mat']; % Construct the filename
    data = load(file_name);
 
    VaR_Values = data.Post_RNNHaR.DataAnneal.VaR_next';
    T = length(data.y);
    retout = data.y_all(T+1:end,:);
    quantile_level = data.Post_RNNHaR.DataAnneal.score.alpha;
    
    Quantile_Score_SMC = data.Post_RNNHaR.DataAnneal.score.Quantile_Score;

    var_rate= ((length(retout(retout<VaR_Values)))/length(retout))/quantile_level;
    
    % include the firm loss
    for k=1:length(retout)
        if retout(k)<VaR_Values(k)
        firm_loss(k)= (retout(k)-VaR_Values(k))^2;
        else
        firm_loss(k)= -quantile_level*retout(k);
        firm_loss(k)= 0;
        end    
    end
    firm_loss_sum= sum(firm_loss);
    %Q_Loss = sum(((retout<VaR_Values)-quantile_level).*(VaR_Values-retout));
    
    [QL1] = varesloss_t(retout,VaR_Values,quantile_level);
     QLoss=[QL1'];
     QuantileLoss = sum(QLoss,1)'; % Column sum
     Quantile_Score = QuantileLoss/length(retout);
     
     %richard code for DQ test
    [dqpV1, dqV1] = dqtest(retout, VaR_Values, quantile_level, 1);
    [dqpV2, dqV2] = dqtest(retout, VaR_Values, quantile_level, 2);
    [dqpV3, dqV3] = dqtest(retout, VaR_Values, quantile_level, 3);
    [dqpV4, dqV4] = dqtest(retout, VaR_Values, quantile_level, 4);
     
    %%%%% VaR Back test
    vbt = varbacktest(retout,VaR_Values,'VaRLevel',[quantile_level]) ;
    TestResults = pof(vbt,'TestLevel',quantile_level);
    vbt.PortfolioID = stock_name;
    vbt.VaRID = 'Normal at 0.025%';
    Backtest = runtests(vbt);% good
    % summary(vbt);
    % pof(vbt);% Kupiec's unconditional coverage test

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Conditional Coverage Test
    num_violations = sum(retout < VaR_Values);
    total_obs = length(retout);
    expected_violations = (quantile_level) * total_obs;

    % Perform Conditional Coverage Test
    if num_violations <= expected_violations
        Cond_Coverage_test = 1; %disp('The VaR model passes the Conditional Coverage Test.');
    else
        Cond_Coverage_test = 0;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Tail Loss Ratioretout
    losses_beyond_VaR = max(0, retout - VaR_Values);
    % total_losses = sum(retout);
    total_losses = 1;

    % Calculate Tail Loss Ratio
    tail_loss_ratio_val = sum(losses_beyond_VaR) / total_losses;
    tail_loss_ratio = num2str(tail_loss_ratio_val);
    % disp(['Tail Loss Ratio: ', num2str(tail_loss_ratio)]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    str = 'ME_New_RNN_HAR_';
    str = append(str,stock_name);
    save(str)

end