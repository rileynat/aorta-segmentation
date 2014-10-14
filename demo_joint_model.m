clc;
clear all;
startup;
alpha = 0.1;
DEBUG = 1;
for i=1:3
     data_preprocess(i, DEBUG); %% (x,y): y=1 for DEBUG MODE
end

%% Run 16 single models first
paramsname_list = cell(16,1);
for i = 1:16
    paramsname_list{i} = sprintf('aorta_split_%02d_%s_itr_%d',i, 'lbfgs', 200);
end

%% Refine segmentation results with 16 learned models
%% pls feel free to replace 'donothing.m' with your own code
framework_train_test_joint(DEBUG, paramsname_list, @donothing, @fit_HUscale, 0, alpha);