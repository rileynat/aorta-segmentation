function [ output_args ] = framework_train_test_joint( DEBUG, paramsname_list, fjoint_model, fnormalize, flag_save, alpha, filterparams)
%FRAMEWORK_TRAIN_TEST_JOINT Summary of this function goes heres
%   Detailed explanation goes here

if ~exist('DEBUG','var')
    DEBUG = 1;
end

if ~exist('flag_save','var'),
    flag_save = 0;
end

if ~exist('alpha','var'),
    alpha = 0.1;
end

if ~exist('fnormalize','var'),
    fnormalize = @fit_HUscale;
end

ORISRC = 'original_data/';
ORISRC_train = [ORISRC 'train/'];
ORISRC_val = [ORISRC 'val/'];
ORISRC_test = [ORISRC 'test/'];

TRAINSET = transformOriData(ORISRC_train, DEBUG, 0);
VALSET = transformOriData(ORISRC_val, DEBUG, 1);
TESTSET = transformOriData(ORISRC_test, DEBUG, 1);

%% Load Models trained separately
MODELSRC = 'models/';
mkdir(MODELSRC);
models = cell(16, 1);
for group_idx = 1:16
    mid = load([MODELSRC sprintf('%s_al%g.mat',paramsname_list{group_idx}, alpha)], 'params', 'weights', 'w', 'b', 'mask_prior');
    models{group_idx} = mid;
end

[VALSET, TESTSET, fname, filterspec] = fjoint_model(DEBUG, TRAINSET, VALSET, TESTSET, models, fnormalize, filterparams);

overall_val_ap = evaluatePerformance(DEBUG, 'val', VALSET, ORISRC, fnormalize, fname, filterspec, flag_save);
overall_test_ap = evaluatePerformance(DEBUG, 'test', TESTSET, ORISRC, fnormalize, fname, filterspec, flag_save);


fprintf('AP: val = %g (std %g), test = %g (std %g)\n',  ...
        mean(overall_val_ap), std(overall_val_ap), mean(overall_test_ap), std(overall_test_ap));
    
if (flag_save),
    fid = fopen(sprintf('log/%s/all_subjects_meanAP.txt',fname), 'a+');
    fprintf(fid, 'Fname: %s \t%s\n',fname, filterspec);
    fprintf(fid, 'AP: val = %g (std %g), test = %g (std %g)\n',  ...
        mean(overall_val_ap), std(overall_val_ap), mean(overall_test_ap), std(overall_test_ap));
    fprintf(fid, '\n');
    fclose(fid);
end

end

%% function evaluatePerformance

function overall_ap = evaluatePerformance(DEBUG, mode, dataset, ORISRC, fnormalize, fname, filterspec, flag_save)

load('global_params.mat');
ORISRC_part = [ORISRC mode '/'];

overall_ap = [];

for i=1:dataset.num,
    xs = dataset.xlist{i};
    ys_pred = dataset.ylist{i};
    seqL = size(xs, 3);
    
    %% Load Ground Truth
    title = ['s' num2str(dataset.tlist(i)) '.mat'];
    DATA = load([ORISRC_part, title]);
    
    ys = [];
    for j = 1:size(DATA.xmap3D, 3),
        if (isnan(DATA.RULER(j))), continue; end
        mask = DATA.ymap3D(:,:,j);
        yc = imresize(mask, [gparams.mean_rsize, gparams.mean_csize], 'bicubic');
        
        idx = round(DATA.RULER(j));
        if (idx < gparams.minv),
            idx = gparams.minv;
        elseif (idx > gparams.maxv),
            idx = gparams.maxv;
        end
        idx = idx - gparams.minv + 1;
        if (idx > 16), break; end
        
        ys = cat(3, ys, yc);
    end
    
    %% Evaluation
    ap_part = [];
    acc_part = [];
    for j = 1:seqL
        
        xc = xs(:,:,j);
        yc = ys(:,:,j);
        yhat = ys_pred(:,:,j);
        xc = fnormalize(xc);
        
        [~,~,ap] = compute_ap(yhat(:),yc(:));
        acc = 1-mean(yc(:) ~= (yhat(:)>0.5));
        ap_part = cat(3, ap_part, ap);
        acc_part = cat(3, acc_part, acc);
        
        if 0, %% visualization
            fig = figure(1);
            set(fig, 'Position', [200, 200, 1200, 220]);
            subplot(1,3,1); imagesc(xc); colormap gray; axis off;
            %title('original image');
            
            subplot(1,3,2); imagesc(yc); colormap gray; axis off;
            %title('ground truth label');
            subplot(1,3,3); imagesc(yhat); colormap gray; axis off;
            %title(sprintf('predicted label, mean AP = %g, studyid = %d', ap, i));
            pause(0.2);
        end
    end
    
    fprintf('subject %d, AP: %s = %g (std %g)\n', dataset.tlist(i), mode, ...
        mean(ap_part), std(ap_part));
    fprintf('subject %d, ACC: %s = %g (std %g)\n', dataset.tlist(i), mode, ...
        mean(acc_part), std(acc_part));
    
    if (flag_save),
        mkdir(sprintf('log/%s',fname));
        fid = fopen(sprintf('log/%s/all_subjects_list_%s.txt',fname,filterspec), 'a+');
        fprintf(fid, '%s\n', title);
        fprintf(fid, 'subject %d, AP: %s = %g (std %g)\n', dataset.tlist(i), mode, ...
            mean(ap_part), std(ap_part));
        fprintf(fid, 'subject %d, ACC: %s = %g (std %g)\n', dataset.tlist(i), mode, ...
            mean(acc_part), std(acc_part));
        fprintf(fid, '\n');
        fclose(fid);
    end
    
    overall_ap = cat(3, overall_ap, ap_part);
    
    if DEBUG,
        break;
    end
    
end

end

%% function transformOriData

function dataset = transformOriData(ORISRC, DEBUG, test_flag)

load('global_params.mat');
%%
OriList = dir([ORISRC '*.mat']);
dataset = struct('num',0,'xlist',[],'ylist',[],'tlist',[]);
dataset.num = length(OriList);
dataset.xlist = cell(dataset.num, 1);
dataset.ylist = cell(dataset.num, 1);
dataset.refzlist = cell(dataset.num, 1); %% referenze group index for z-coordinate
dataset.tlist = zeros(dataset.num, 1);

for i = 1:length(OriList),
    title = OriList(i).name;
    DATA = load([ORISRC title]);
    seqL = size(DATA.xmap3D, 3);
    
    x = [];
    y = [];
    refz = [];
    
    for j = 1:seqL
        if (isnan(DATA.RULER(j))), continue; end
        img = DATA.xmap3D(:,:,j);
        if (test_flag),
            mask = zeros(size(DATA.xmap3D(:,:,j)));
        else
            mask = DATA.ymap3D(:,:,j);
        end
        
        xc = imresize(img, [gparams.mean_rsize, gparams.mean_csize], 'bicubic');
        yc = imresize(mask, [gparams.mean_rsize, gparams.mean_csize], 'bicubic');
        idx = round(DATA.RULER(j));
        if (idx < gparams.minv),
            idx = gparams.minv;
        elseif (idx > gparams.maxv),
            idx = gparams.maxv;
        end
        
        idx = idx - gparams.minv + 1;
        
        if (idx > 16), break; end
        
        %xc = fnormalize(xc);
        x = cat(3, x, xc);
        y = cat(3, y, yc);
        refz = cat(1, refz, idx);
    end
    
    dataset.xlist{i} = x;
    dataset.ylist{i} = y;
    dataset.refzlist{i} = refz;
    dataset.tlist(i) = str2num(title(2:end-4));
    
    if (DEBUG==1),
        break;
    end
end
end