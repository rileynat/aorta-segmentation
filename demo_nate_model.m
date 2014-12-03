
close all;
clc;
startup;

addpath('data');

for i=1:3
     data_preprocess( i, 0 ); %% (x,y): y=1 for DEBUG MODE
end


train_data_x = [];
train_data_y = [];


for i=4
    filename = sprintf('normalized_data/train/%02d.mat', i);
    cModel = load(filename);
    x_input = double(cModel.cModel.x);
    y_input = double(cModel.cModel.y);
    train_data_x = cat(3, train_data_x, x_input);
    train_data_y = cat(3, train_data_y, y_input);
end

val_data_x = [];
val_data_y = [];

for i=4
    filename = sprintf('normalized_data/val/%02d.mat', i);
    cModel = load(filename);
    x_input = double(cModel.cModel.x);
    y_input = double(cModel.cModel.y);
    val_data_x = cat(3, val_data_x, x_input);
    val_data_y = cat(3, val_data_y, y_input);
end

test_data_x = [];
test_data_y = [];

for i=4
    filename = sprintf('normalized_data/test/%02d.mat', i);
    cModel = load(filename);
    x_input = double(cModel.cModel.x);
    y_input = double(cModel.cModel.y);
    test_data_x = cat(3, test_data_x, x_input);
    test_data_y = cat(3, test_data_y, y_input);
end

filterInfo = struct;
filterInfo.numFilters1 = 30;
filterInfo.filterSize1 = 8;
filterInfo.numFilters2 = 30;
filterInfo.filterSize2 = 16;
filterInfo.numFilters3 = filterInfo.numFilters2;
filterInfo.filterSize3 = filterInfo.filterSize1 + filterInfo.filterSize2 - 1;

addpath('utils/');
train_data_x = fit_HUscale(train_data_x);

addpath('cnn/');

[weights] = train_cnn(train_data_x(:,:,1:1), train_data_y(:,:,1:1), filterInfo); 

save(strcat('cnn/weights/weights_', date, '.mat'), 'weights');

[validAcc] = validate_cnn(val_data_x(:,:,:), val_data_y(:,:,:), weights, filterInfo);

disp(validAcc);

[testAcc] = test_cnn(test_data_x(:,:,:), test_data_y(:,:,:), weights, filterInfo);

disp(testAcc);
