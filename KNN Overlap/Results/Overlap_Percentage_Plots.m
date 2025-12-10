width = 500;
height = 700;
x_tick_font_size = 20;
y_tick_font_size = 20;
x_label_font_size = 26;
y_label_font_size = 26;

%%
load('Uniform_DBLP_KNN_Overlap_Percentages.mat')

fig = figure;
set(fig, 'Position', [100, 100, width, height]);

b = bar(store_overlap_percentage,0.25);

colors = [
    1 0 0;   % Red
    0 1 0;   % Green
];

b.FaceColor = 'flat';    % Use flat coloring
b.CData = colors;

xticklabels({'k_l-1', 'k_l', 'k_l+1', 'k_l+2', 'k_l+3'});

yticklabels({'0%','', '20%', '', '40%', '', '60%', '70%', '80%', '90%', '100%'});

xlabel('Nearest Neighbors','FontName', 'Times New Roman', 'FontSize', x_label_font_size, 'FontWeight', 'bold')
ylabel('Overlap Percentage','FontName', 'Times New Roman', 'FontSize', y_label_font_size, 'FontWeight', 'bold')
ylim([60,100])

ax = gca; % Get current axes
ax.XAxis.FontName = 'Times New Roman';  % Set font for x-axis ticks
ax.XAxis.FontSize = x_tick_font_size;           % Set font size for x-axis ticks
ax.YAxis.FontName = 'Times New Roman';  % Set font for x-axis ticks
ax.YAxis.FontSize = y_tick_font_size;           % Set font size for x-axis ticks

%%
load('Non_Uniform_DBLP_KNN_Overlap_Percentages.mat')
fig = figure;
set(fig, 'Position', [100, 100, width, height]);

b = bar(store_overlap_percentage,0.45);

colors = [
    1 0 0;   % Red
    1 0 0;   % Red
    1 0 0;   % Red
    1 0 0;   % Red
    0 1 0;   % Green
];

b.FaceColor = 'flat';    % Use flat coloring
b.CData = colors;

xticklabels({'k_l-1', 'k_l', 'k_l+1', 'k_l+2', 'k_l+3'});

yticklabels({'0%','', '20%', '', '40%', '', '60%', '70%', '80%', '90%', '100%'});

xlabel('Nearest Neighbors','FontName', 'Times New Roman', 'FontSize', x_label_font_size, 'FontWeight', 'bold')
ylabel('Overlap Percentage','FontName', 'Times New Roman', 'FontSize', y_label_font_size, 'FontWeight', 'bold')
ylim([60,100])

ax = gca; % Get current axes
ax.XAxis.FontName = 'Times New Roman';  % Set font for x-axis ticks
ax.XAxis.FontSize = x_tick_font_size;           % Set font size for x-axis ticks
ax.YAxis.FontName = 'Times New Roman';  % Set font for x-axis ticks
ax.YAxis.FontSize = y_tick_font_size;           % Set font size for x-axis ticks


%%
load('Uniform_Cora_KNN_Overlap_Percentages.mat')
fig = figure;
set(fig, 'Position', [100, 100, width, height]);

b = bar(store_overlap_percentage,0.25);

colors = [
    1 0 0;   % Red
    0 1 0;   % Green
];

b.FaceColor = 'flat';    % Use flat coloring
b.CData = colors;

xticklabels({'k_l-1', 'k_l', 'k_l+1', 'k_l+2', 'k_l+3'});

yticklabels({'0%','', '20%', '', '40%', '', '60%', '70%', '80%', '90%', '100%'});

xlabel('Nearest Neighbors','FontName', 'Times New Roman', 'FontSize', x_label_font_size, 'FontWeight', 'bold')
ylabel('Overlap Percentage','FontName', 'Times New Roman', 'FontSize', y_label_font_size, 'FontWeight', 'bold')
ylim([60,100])

ax = gca; % Get current axes
ax.XAxis.FontName = 'Times New Roman';  % Set font for x-axis ticks
ax.XAxis.FontSize = x_tick_font_size;           % Set font size for x-axis ticks
ax.YAxis.FontName = 'Times New Roman';  % Set font for x-axis ticks
ax.YAxis.FontSize = y_tick_font_size;           % Set font size for x-axis ticks

%%
load('Non_Uniform_Cora_KNN_Overlap_Percentages.mat')
fig = figure;
set(fig, 'Position', [100, 100, width, height]);

b = bar(store_overlap_percentage,0.45);

colors = [
    1 0 0;   % Red
    1 0 0;   % Red
    1 0 0;   % Red
    1 0 0;   % Red
    0 1 0;   % Green
];

b.FaceColor = 'flat';    % Use flat coloring
b.CData = colors;

xticklabels({'k_l-1', 'k_l', 'k_l+1', 'k_l+2', 'k_l+3'});

yticklabels({'0%','', '20%', '', '40%', '', '60%', '70%', '80%', '90%', '100%'});

xlabel('Nearest Neighbors','FontName', 'Times New Roman', 'FontSize', x_label_font_size, 'FontWeight', 'bold')
ylabel('Overlap Percentage','FontName', 'Times New Roman', 'FontSize', y_label_font_size, 'FontWeight', 'bold')
ylim([60,100])

ax = gca; % Get current axes
ax.XAxis.FontName = 'Times New Roman';  % Set font for x-axis ticks
ax.XAxis.FontSize = x_tick_font_size;           % Set font size for x-axis ticks
ax.YAxis.FontName = 'Times New Roman';  % Set font for x-axis ticks
ax.YAxis.FontSize = y_tick_font_size;           % Set font size for x-axis ticks