%% Run to automatically set up necessary paths

% Get current working directory
basePath = pwd;

% Relative to the base path, create folder names
folder1 = fullfile(basePath, 'Algorithms');
folder2 = fullfile(basePath, 'Datasets');
folder3 = fullfile(basePath, 'Experiments');
folder4 = fullfile(basePath, 'Functions');
folder5 = fullfile(basePath, 'KNN Overlap');
folder6 = fullfile(basePath, 'Total Variations');

% Add the folders to the MATLAB path
addpath(genpath(folder1));
addpath(genpath(folder2));
addpath(genpath(folder3));
addpath(genpath(folder4));
addpath(genpath(folder5));
addpath(genpath(folder6));

% Save the path changes. If you do not want to save these paths for future use, comment out this line
savepath;

disp('Successfully added paths');