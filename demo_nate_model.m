close all;
clc;
startup;

for i=1:3
     data_preprocess( i, 0 ); %% (x,y): y=1 for DEBUG MODE
end


train_data_x = [];
train_data_y = [];
%train_data_t = [];

for i=1:17
    filename = sprintf('normalized_data/train/%02d.mat', i);
    cModel = load(filename);
    x_input = double(cModel.cModel.x);
    y_input = double(cModel.cModel.y);
    train_data_x = cat(3, train_data_x, x_input);
    train_data_y = cat(3, train_data_y, y_input);
end

numFilters = 50;
filterSize = 16;

addpath('utils/');
train_data_x = fit_HUscale(train_data_x);

addpath('nate_cnn/');

[inToHidFilters, inToHidBias, hidToOutFilters, hidToOutBias] = train_cnn(train_data_x, train_data_y, numFilters, filterSize); 
