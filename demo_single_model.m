clc;
clear all;
startup;
for i=1:3
     data_preprocess( i, 1 ); %% (x,y): y=1 for DEBUG MODE
end
%% Preprocessing step: normalized CT scans are saved in the folder called "normalized_data/"

framework_train_test_sep( 0, 1, @train_weights_1layercnn_sep , @fit_HUscale, 0, 0.1, 1);
%% TODO: implement another function for segmenting aorta in each group
%% 1) pls feel free to modify 'train_weights_1layercnn_sep.m'
%% or even write your own model for aorta segmentation (group-based)
%% 2) pls also take a look at the folder "vis/" and "log/",
%% where visualization results and meanAP (overall performance) are stored
