function score = one_step_forecast_score(VaR_next,y_next,score)

alpha_sig = score.alpha;

score.qs = score.qs + (y_next-VaR_next)*(alpha_sig-utils_indicator_fun(y_next,VaR_next));

end

