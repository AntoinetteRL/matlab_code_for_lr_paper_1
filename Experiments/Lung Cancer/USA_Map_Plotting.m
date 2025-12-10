function states = USA_Map_Plotting(state_vals)

% Define data for each state
% (Order the values according to the alphabetical order of state names)

% Load state boundary data
states = shaperead('usastatelo', 'UseGeoCoords', true);
states([2, 11, 51],:) = [];

% Create a map
fig = figure;

set(fig, 'Position', [100, 100, 1000, 600]);

usamap('conus');
% setm(ax, 'FFaceColor', [0.9 0.9 0.9]); % Set background color

% Define a colormap and normalize data for coloring
colormap(jet); % Choose your desired colormap
% clim([min(state_vals) max(state_vals)]); % Scale the colorbar
clim([-2 2]); % Scale the colorbar
c = colorbar;
set(c, 'FontSize', 22);

% Loop through each state and plot it
for k = 1:length(states)
    stateName = states(k).Name;
    % Find the index of the state in your data
    stateIndex = find(strcmpi(stateName, {states.Name}));
    if ~isempty(stateIndex)
        % Get the value for the current state
        value = state_vals(stateIndex);
        
        % Determine the color from the colormap
        color = interp1(linspace(min(state_vals), max(state_vals), 256), ...
                        colormap, value, 'linear');
        
        % Plot the state with the color
        patchm(states(k).Lat, states(k).Lon, color);
    end
end

framem off
gridm off
mlabel off
plabel off

end