function smoking_data = Smoking_Rates_Data_Processing()

%% Read Smoking Rates
data = readtable('Smoking Rates.csv');

data(strcmp(data{:,"LocationDescription"}, 'District of Columbia'), :) = [];
data(strcmp(data{:,"LocationDescription"}, 'Alaska'), :) = [];
data(strcmp(data{:,"LocationDescription"}, 'Hawaii'), :) = [];

smoking_data = [];
for year = 2011:2019
    temp = data(data{:,"Year"} == year, [1,2,8]);
    smoking_data(:,end+1) = temp{:,"DataValue"};
end

% Estimate the NaN value with a linear fitted model
coef = polyfit(1:8,smoking_data(28,1:end-1),1);
smoking_data(28,9) = polyval(coef,9);

end