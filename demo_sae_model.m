close all;
clc;
startup;

addpath('data');
addpath('common');
addpath('sparse_ae');

for i=1:3
     data_preprocess( i, 0 ); %% (x,y): y=1 for DEBUG MODE
end


temp_train_data_x = [];
temp_train_data_y = [];


for i=4
    filename = sprintf('normalized_data/train/%02d.mat', i);
    cModel = load(filename);
    x_input = double(cModel.cModel.x);
    y_input = double(cModel.cModel.y);
    temp_train_data_x = cat(3, temp_train_data_x, x_input);
    temp_train_data_y = cat(3, temp_train_data_y, y_input);
end

temp_val_data_x = [];
temp_val_data_y = [];

for i=4
    filename = sprintf('normalized_data/val/%02d.mat', i);
    cModel = load(filename);
    x_input = double(cModel.cModel.x);
    y_input = double(cModel.cModel.y);
    temp_val_data_x = cat(3, temp_val_data_x, x_input);
    temp_val_data_y = cat(3, temp_val_data_y, y_input);
end

temp_test_data_x = [];
temp_test_data_y = [];

for i=4
    filename = sprintf('normalized_data/test/%02d.mat', i);
    cModel = load(filename);
    x_input = double(cModel.cModel.x);
    y_input = double(cModel.cModel.y);
    temp_test_data_x = cat(3, temp_test_data_x, x_input);
    temp_test_data_y = cat(3, temp_test_data_y, y_input);
end

%%% Make all of the input Data Square
addpath('common/');
train_data_x = make_square(temp_train_data_x);
train_data_y = make_square(temp_train_data_y);
val_data_x = make_square(temp_val_data_x);
val_data_y = make_square(temp_val_data_y);
test_data_x = make_square(temp_test_data_x);
test_data_y = make_square(temp_test_data_y);

addpath('utils/');
train_data_x = fit_HUscale(train_data_x);

% Train Autoencoder
subset_size = 3;
size(train_data_x)
 


train_x_roll_subset = reshape(train_data_x(:,:,1:subset_size), size(train_data_x, 1) * ...
    size(square_train_data_x, 2), subset_size);

reduction_factor = 4/5;
reduction = size(train_data_x, 1) - size(train_data_x, 1)*reduction_factor;
saeInfo = struct;
saeInfo.inputSize = size(train_x_roll_subset, 1);
saeInfo.hiddenSize = saeInfo.inputSize * reduction_factor; %CHANGE THIS
saeInfo.sparsity = .1;
saeInfo.lambda = 3e-3;
saeInfo.beta = 3;

input_features = RunSparseAE(train_x_roll_subset, saeInfo);
save('input_features');


% Reshape Features
cnn_input = reshape(input_features, size(train_data_x, 1)*reduction_factor, ...
    size(train_data_x, 2)*reduction_factor, subset_size);


disp('past autoencoder stuff');

filterInfo = struct;
filterInfo.numFilters1 = 30;
filterInfo.filterSize1 = 5;
filterInfo.numFilters2 = 30;
filterInfo.filterSize2 = 16;
filterInfo.numFilters3 = filterInfo.numFilters2;
filterInfo.filterSize3 = filterInfo.filterSize1 + filterInfo.filterSize2 + reduction - 1;

addpath('cnn/');

[weights] = train_cnn(train_data_x(:,:,:), train_data_y(:,:,:), filterInfo); 

save(strcat('nate_cnn/weights/weights_', date, '.mat'), 'weights');

[validAcc] = validate_cnn(val_data_x(:,:,:), val_data_y(:,:,:), weights, filterInfo);

disp(validAcc);

[testAcc] = test_cnn(test_data_x(:,:,:), test_data_y(:,:,:), weights, filterInfo);

disp(testAcc);
