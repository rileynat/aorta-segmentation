function [ output_args ] = framework_train_test_joint( DEBUG, paramsname_list, fjoint_model, fnormalize, flag_save, alpha)
%FRAMEWORK_TRAIN_TEST_JOINT Summary of this function goes heres
%   Detailed explanation goes here
load('global_params.mat');
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
VALSET = transformOriData(ORISRC_val, DEBUG, 0);
TESTSET = transformOriData(ORISRC_test, DEBUG, 1);

%% Load Models trained separately
MODELSRC = 'models/';
mkdir(MODELSRC);
models = cell(16, 1);
for group_idx = 1:16
    mid = load([MODELSRC sprintf('%s_al%g.mat',paramsname_list{group_idx}, alpha)], 'params', 'weights', 'w', 'b', 'mask_prior');
    models{group_idx} = mid;
end

[TESTSET, fname] = fjoint_model(DEBUG, models, TRAINSET, VALSET, TESTSET, fnormalize, alpha);

overall_ap = [];

for i=1:TESTSET.num,
    xs = TESTSET.xlist{i};
    ys_pred = TESTSET.ylist{i};
    seqL = size(xs, 3);
    
    %% Load Ground Truth
    title = ['s' num2str(TESTSET.tlist(i)) '.mat'];
    DATA = load([ORISRC_test, title]);
    
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
    ap_test = [];
    acc_test = [];
    for j = 1:seqL
        
        xc = xs(:,:,j);
        yc = ys(:,:,j);
        yhat = ys_pred(:,:,j);
        xc = fnormalize(xc);
        
        [~,~,ap] = compute_ap(yhat(:),yc(:));
        acc = 1-mean(yc(:) ~= (yhat(:)>0.5));
        ap_test = cat(3, ap_test, ap);
        acc_test = cat(3, acc_test, acc);
        
        if 1, %% visualization
            fig = figure(1);
            set(fig, 'Position', [200, 200, 1200, 220]);
            subplot(1,3,1); imagesc(xc); colormap gray; axis off;
            %title('original image');
            
            subplot(1,3,2); imagesc(yc); colormap gray; axis off;
            %title('ground truth label');
            subplot(1,3,3); imagesc(yhat); colormap gray; axis off;
            %title(sprintf('predicted label, mean AP = %g, studyid = %d', ap, i));
            pause;
        end
    end
    
    fprintf('subject %d, AP: test = %g (std %g)\n', TESTSET.tlist(i), ...
        mean(ap_test), std(ap_test));
    fprintf('subject %d, ACC: test = %g (std %g)\n', TESTSET.tlist(i), ...
        mean(acc_test), std(acc_test));
    
    if (flag_save),
        mkdir(sprintf('log/%s',fname));
        fid = fopen(sprintf('log/%s/all_subjects_list.txt',fname), 'a+');
        fprintf(fid, '%s\n', title);
        fprintf(fid, 'subject %d, AP: test = %g (std %g)\n', TESTSET.tlist(i), ...
            mean(ap_test), std(ap_test));
        fprintf(fid, 'subject %d, ACC: test = %g (std %g)\n', TESTSET.tlist(i), ...
            mean(acc_test), std(acc_test));
        fprintf(fid, '\n');
        fclose(fid);
    end
    
    overall_ap = cat(3, overall_ap, ap_test);
    
    if DEBUG,
        break;
    end
    
end

fprintf('AP: test = %g (std %g)\n',  ...
        mean(overall_ap), std(overall_ap));
    
if (flag_save),
    fid = fopen(sprintf('log/%s/all_subjects_meanAP.txt',fname), 'w');
    fprintf(fid, 'AP: test = %g (std %g)\n',  ...
        mean(overall_ap), std(overall_ap));
    fprintf(fid, '\n');
    fclose(fid);
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