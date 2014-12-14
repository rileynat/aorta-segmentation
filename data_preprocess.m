function [ output_args ] = data_preprocess( MODE, DEBUG )
%DATA_NORMALIZATION Summary of this function goes here
%   Detailed explanation goes here

if ~exist('maxsize', 'var'),
    maxsize = 70;
end

if ~exist('mean_rsize','var'),
    mean_rsize = 70;
end

if ~exist('mean_csize','var'),
    mean_csize = 105;
end

if ~exist('MODE', 'var'),
    MODE = 1; % 1 for 'train', 2 for 'valid', 3 for 'test'
end

if ~exist('DEBUG', 'var'),
    DEBUG = 1;
end

if (MODE == 1),
    modename = 'train/';
elseif (MODE == 2),
    modename = 'val/';
elseif (MODE == 3),
    modename = 'test/';
end

ORISRC = ['original_data/' modename];
NORMSRC = ['normalized_data/' modename];

rsize = [];
csize = [];

addpath('/mnt/neocortex3/scratch/xcyan/aorta_seg/');

%RL = load('VLCrulers.mat');
mkdir(ORISRC);
mkdir(NORMSRC);

%% Normailizing # of rows and # of cols
if (length(dir([ORISRC '*.mat'])) > 0)
    fprintf('Original data already exists\n');
    load('global_params.mat');
else
    fprintf('Loading original data...\n');
    gparams = struct('mean_rsize',mean_rsize,'mean_csize',mean_csize);
    for fnum = 1:6
        %% Train-Val-Test Splits
        if (MODE == 1 && ~(fnum <=3 && fnum >= 1))
            continue;
        elseif (MODE == 2 && ~(fnum == 4))
            continue;
        elseif (MODE == 3 && ~(fnum >= 5 && fnum <= 6))
            continue;
        end
        
        fileName = sprintf('vasc_GS_%d.mat',fnum);
        GSlist = whos('-file',fileName);
        
        for i = 1:length(GSlist),
            GSdata = load(fileName, GSlist(i).name);
            DATA = GSdata.(GSlist(i).name);
            RULER = RL.VLCrulers.(GSlist(i).name);  %% 1*H vector
            %%
            xmap3D = DATA.VOX.V;	%% R*C*H matrix
            ymap3D = DATA.BW;		%% R*C*H binary matrix
            coordX = DATA.VOX.xVec;	%% 1*C vector
            coordY = DATA.VOX.yVec; %% 1*R vector
            coordZ = DATA.VOX.zVec; %% 1*H vector
            
            save([ORISRC GSlist(i).name '.mat'],'xmap3D','ymap3D','coordX','coordY','coordZ','RULER');
            
            rsize = cat(1, rsize, length(DATA.VOX.yVec));
            csize = cat(1, csize, length(DATA.VOX.xVec));
            if (DEBUG), break; end
        end
        if (DEBUG), break; end
        
    end
    %ratio = rsize./csize;
    %gparams.mean_rsize = min(maxsize/14, round(mean(rsize)/14))*14;
    %gparams.mean_csize = round(gparams.mean_rsize./mean(ratio)/14)*14;
    
    save('global_params.mat','gparams');
    
end


%% Decide # of models based on referencing heights

OriList = dir([ORISRC,'*.mat']);
% maxv = -1e5;
% minv = 1e5;
% for i = 1:length(OriList),
%     title = OriList(i).name;
%     DATA = load([ORISRC title]);
%     maxv = max(maxv, max(DATA.RULER));
%     minv = min(minv, min(DATA.RULER));
% end

%% reference number 0..17 (GS_1 ~ GS_6)
if (length(dir([NORMSRC '*.mat'])) > 0)
    fprintf('Normalized data already exists\n');
    load('global_params.mat');
else
    fprintf('Producing normalized data...\n');
    
    gparams = setfield(gparams, 'minv', 0);
    gparams = setfield(gparams, 'maxv', 17);
    
    models = struct('num', 0, 'idx', [], 'xlist', [], 'ylist', [], 'tlist', []);
    models.num = gparams.maxv - gparams.minv + 1;
    models.idx = gparams.minv:gparams.maxv;
    models.xlist = cell(models.num, 1);
    models.ylist = cell(models.num, 1);
    models.tlist = cell(models.num, 1);
    
    for i = 1:length(OriList),
        title = OriList(i).name;
        DATA = load([ORISRC title]);
        seqL = size(DATA.xmap3D, 3);
        
        for j = 1:seqL,
            if (isnan(DATA.RULER(j))), continue; end
            img = DATA.xmap3D(:,:,j);
            mask = DATA.ymap3D(:,:,j);
            %%
            img = imresize(img, [gparams.mean_rsize, gparams.mean_csize], 'bicubic');
            mask = imresize(mask, [gparams.mean_rsize, gparams.mean_csize], 'bicubic');
            idx = round(DATA.RULER(j));
            if (idx < gparams.minv),
                idx = gparams.minv;
            elseif (idx > gparams.maxv),
                idx = gparams.maxv;
            end
            
            idx = idx - gparams.minv + 1;
            
            models.xlist{idx} = cat(3, models.xlist{idx}, img);
            models.ylist{idx} = cat(3, models.ylist{idx}, mask);
            % 's123.mat' --> 123
            models.tlist{idx} = cat(1, models.tlist{idx}, [str2num(title(2:end-4)), j]);
            
        end
        
        for k = 1:models.num
            fprintf('%d ', size(models.xlist{k}, 3));
        end
        fprintf('\n');
        
    end
    
    for i = 1:models.num
        cModel = struct('x', [], 'y', [], 't', []);
        cModel.x = models.xlist{i}; %% mean_rsize * mean_csize * #samples
        cModel.y = models.ylist{i}; %% mean_rsize * mean_csize * #samples
        cModel.t = models.tlist{i}; %% #samples * 2: idx of raw data ('s1004'), idx of height
        
        save([NORMSRC sprintf('%02d', i) '.mat'],'cModel');
    end
    save('global_params.mat','gparams');
end

end
