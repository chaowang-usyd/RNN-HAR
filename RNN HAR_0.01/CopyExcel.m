
stock_name_list = {'AEX','AORD','BFX','BSESN','BVLG','BVSP','DJI','FCHI','FTMIB','FTSE',...
    'GDAXI','GSPTSE','HSI','IBEX','IXIC','KS11','KSE','MXX','N225','NSEI','OMXC20','OMXHPI','OMXSPI',...
    'OSEAX','RUT','SMSI','SPX','SSEC','SSMI','STI','STOXX50E'};

num_values = 12;
data_cell = cell(length(stock_name_list), num_values);


% Define column labels
column_labels = {'Stock Name', 'Quantile Score', 'Var Rate', 'dqpV1', 'dqpV2', ...
    'dqpV3', 'dqpV4', 'POF', 'CC', 'Cond Coverage Test', ...
    'Tail Loss Ratio','Firm Loss'};

% Add column labels to the first row of data_cell
data_cell(1, :) = column_labels;

for j = 1:length(stock_name_list)
    stock_name = stock_name_list{j};       
    disp('=========== stock name =====================')
    stock_name
    disp('==============================================')
    
      %% prepare data
    file_name = ['ME_New_RNN_HAR_' stock_name '.mat']; % Construct the filename
    data = load(file_name);

    CC_value = char(data.Backtest.CC);
    POF_value = char(data.Backtest.POF);
    % Initialize variable to store result

    CC_result = zeros(size(CC_value));
    POF_result = zeros(size(POF_value));
    
    if strcmp(CC_value, 'reject')
        CC_result = 1;
    else
        CC_result = 0;
    end


    if strcmp(POF_value, 'reject')
        POF_result = 1;
    else
        POF_result = 0;
    end

    filename = ['Results', '.csv']; % Construct filename
    data_cell{j+1, 1} = data.stock_name;
    data_cell{j+1, 2} = data.Quantile_Score;
    data_cell{j+1, 3} = data.var_rate;
    data_cell{j+1, 4} = data.dqpV1;
    data_cell{j+1, 5} = data.dqpV2;
    data_cell{j+1, 6} = data.dqpV3;
    data_cell{j+1, 7} = data.dqpV4;
    data_cell{j+1, 8} = POF_result;
    data_cell{j+1, 9} = CC_result;
    data_cell{j+1, 10} = data.Cond_Coverage_test;
    data_cell{j+1, 11} = data.tail_loss_ratio;
    data_cell{j+1, 12} = data.firm_loss_sum;    

end

    xlswrite('ProposedRNNHAR1.xlsx', data_cell);
