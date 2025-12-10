function temperature_data = Temperature_Data_Processing()

temperature_data = zeros(48,9);  % 48 states, 9 years

idx = 1;
for year = 2011:2019
    data_string = ['Avg_Temp_', num2str(year), '.csv'];
    data = readtable(data_string);  % Read .csv file
    temperature_data(:,idx) = data{:,'Value'};
    idx = idx + 1;
end

end