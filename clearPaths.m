%% Run to automatically remove paths

% Get current working directory
basePath = pwd;

% Relative to the base path, create folder names
folder1 = fullfile(basePath, 'Algorithms');
folder2 = fullfile(basePath, 'Datasets');
folder3 = fullfile(basePath, 'Experiments');
folder4 = fullfile(basePath, 'Functions');
folder5 = fullfile(basePath, 'KNN Overlap');
folder6 = fullfile(basePath, 'Total Variations');

% Remove folders from the MATLAB path
rmpath(genpath(folder1));
rmpath(genpath(folder2));
rmpath(genpath(folder3));
rmpath(genpath(folder4));
rmpath(genpath(folder5));
rmpath(genpath(folder6));

% Save the path removals. If you do not want to save these changes, comment out this line
savepath;

disp('Successfully removed paths');