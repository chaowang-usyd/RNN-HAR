% Initialize arrays to store Monte Carlo errors for each parameter
clc;clear;
numParticles = 2000;% Number of particles
numParameters = 15; % Number of parameters
numRuns = 10; % Number of runs
monteCarloErrorsLA = zeros(numParameters, numRuns);
monteCarloErrorsDA = zeros(numParameters, numRuns);
monteCarloErrorsLogLike = zeros(numRuns,1);
for run = 1:numRuns
    % Construct the variable name dynamically based on the run index
    variableName = sprintf('Results_RNN_HAR_AEX', run);
    try
        % Load the data from the corresponding file
        load([variableName '.mat']);  % Load Results_DeepHAR_SMC2noUP1_AEX.mat etc
    catch
        error(['Error loading file: ' variableName '.mat. Make sure the file exists.']);
    end
    % Access the LikAnneal structure and DataAnneal structure
    try
        likStruct = Post_RNNHaR.LikAnneal;
        DatStruct = Post_RNNHaR.DataAnneal;
    catch
        error(['Error accessing LikAnneal structure in ' variableName]);
        error(['Error accessing DataAnneal structure in ' variableName]);
    end
    % Extract parameter values for each parameter
    for paramIdx = 1:numParameters
        % Assuming each parameter has fields like alpha_f1_d, alpha_f2_d, b_f_d, etc.
         fieldNames = {'alpha_d_0', 'alpha_d_1', 'alpha_d_2', 'alpha_w_0', 'alpha_w_1', 'alpha_w_2', 'alpha_m_0', 'alpha_m_1', 'alpha_m_2', ...
            'beta0','beta1','beta2','beta3','beta4','sigmaLH'};  % Add all relevant field names
        fieldName = fieldNames{paramIdx};
       % Extract values for the current parameter
        parameterValuesLA = likStruct.(fieldName);
        parameterValuesDA = DatStruct.(fieldName);
        LogLikeValues = likStruct.('log_llh');
        QuantileScore = DatStruct.score.('Quantile_Score');

        % Calculate the Monte Carlo error for the current parameter
        monteCarloErrorsLA(paramIdx, run) = std(parameterValuesLA) ;%/ sqrt(numParticles);
        monteCarloErrorsDA(paramIdx, run) = std(parameterValuesDA);% / sqrt(numParticles);
        monteCarloErrorsLogLike(run,1)  =  LogLikeValues;%LogLikeValues;  
        monteCarloErrorsQS(run,1) = QuantileScore;

    end
end
% Calculate the mean Monte Carlo error across runs for each parameter
monteCarloErrorsMeanLA = mean(monteCarloErrorsLA, 2);
monteCarloErrorsMeanDA = mean(monteCarloErrorsDA, 2);
monteCarloErrorsMeanLogLH = mean(monteCarloErrorsLogLike);
monteCarloErrorsMeanQS = mean(monteCarloErrorsQS);
% Calculate the Monte Carlo standard error across runs for each parameter
monteCarloErrorsSELA = std(monteCarloErrorsLA, 0, 2);
monteCarloErrorsSEDA = std(monteCarloErrorsDA, 0, 2);
monteCarloErrorsSELogLH = std(monteCarloErrorsLogLike);%/sqrt(numRuns);
monteCarloErrorsSEQS = std(monteCarloErrorsQS);
% Display or use the results as needed

save FTSE_RNNHARMCnormal1con;