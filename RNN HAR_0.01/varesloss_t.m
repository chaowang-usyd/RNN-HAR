function [QuntileLoss] = varesloss_t(y,VaR,alp)

n=length(y);
QuntileLoss=0;

for t=1:n

    QuntileLoss(t)=((y(t)<VaR(t))-alp)*(VaR(t)-y(t));

end  

% % use the negative of AL likelihood as loss function
% 
% l_re_EScavair_1= log((alp-1)./ES);
% l_re_EScavair_2= ((y-VaR).*(alp-(y<=VaR)))./(alp*ES);
% VaRESJointLoss= -(l_re_EScavair_1+l_re_EScavair_2)';  