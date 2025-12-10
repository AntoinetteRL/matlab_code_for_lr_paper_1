function mortality_data = Lung_Cancer_Data_Processing()

%% Read Mortality Rate Data
data1 = readtable('Lung Cancer Mortality Rates.xlsx');

nan_idx = isnan(data1{1,2:end});
data1(:,[false, nan_idx]) = [];  % Each column with the year is the total mortality rate. So we remove the other columns (they are NaN)
data1([2, end-1, end],:) = [];  % Uneccesary rows of NaNs
data1(strcmp(data1{:,1}, 'U.S. Total'),:) = [];  % Delete row of totals
data1(strcmp(data1{:,1}, 'District of Columbia'),:) = [];
data1(strcmp(data1{:,1}, 'Alaska'),:) = [];
data1(strcmp(data1{:,1}, 'Hawaii'),:) = [];

raw_mortality_data = data1{2:end, 2:end};  % Each row is a state (node) and each column is an observation (year). Mortality rate is persons per 100,000. Includes all years
mortality_data = raw_mortality_data(:,9:17);  % Isolates years corresponding to the years for smoking rates (2011 - 2019)

end