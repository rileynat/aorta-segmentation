function [paramsname] = framework_train_test_submodel( DEBUG, group_idx_range, flocal_model, fnormalize, flag_save, alpha, flag_visualize )
%TRAIN_WEIGHTS_CNN_SEP Summary of this function goes here
%   Detailed explanation goes here
%   Usage:
%   framework_train_test_sep(1, 1, @train_weights_1layercnn_sep, 0);
%   framework_train_test_sep(0, 1:18, @train_weights_1layercnn_sep, 1);

if ~exist('DEBUG', 'var'),
    DEBUG = 1;
end

if ~exist('group_idx_range', 'var'),
    group_idx_range = 1;
end

if ~exist('flag_save','var'),
    flag_save = 0;
end

if ~exist('flag_visualize','var')
    flag_visualize = 0;
end

if ~exist('alpha', 'var'),
    % masking hyperparameter
    % larger the alpha, mask gets close to step function
    alpha = 0.1;
end

if ~exist('fnormalize', 'var'),
    fnormalize = @fit_HUscale;
end


NORMSRC = 'normalized_data/';
NORMSRC_train = [NORMSRC 'train/'];
NORMSRC_val = [NORMSRC 'val/'];
NORMSRC_test = [NORMSRC 'test/'];

for group_idx = group_idx_range,
    TRAINSET = load([NORMSRC_train sprintf('%02d',group_idx) '.mat'], 'cModel');
    VALSET = load([NORMSRC_val sprintf('%02d',group_idx) '.mat'], 'cModel');
    TESTSET = load([NORMSRC_test sprintf('%02d',group_idx) '.mat'], 'cModel');
    
    xtrain = TRAINSET.cModel.x;
    ytrain = TRAINSET.cModel.y;
    xval = VALSET.cModel.x;
    yval = VALSET.cModel.y;
    xtest = TESTSET.cModel.x;
    ytest = TESTSET.cModel.y;
    
    if (DEBUG),
        xtrain = xtrain(:, :, 1:10);
        ytrain = ytrain(:, :, 1:10);
        xval = xval(:, :, 1:10);
        yval = yval(:, :, 1:10);
        xtest = xtest(:, :, 1:10);
        ytest = ytest(:, :, 1:10);
    end
    
    %% Normalize Values
    xtrain = fnormalize(xtrain);
    xval = fnormalize(xval);
    xtest = fnormalize(xtest);
    
    %% Apply Local models
    [yval_pred, ytest_pred, inf_time, paramsname] = flocal_model(group_idx, xtrain, ytrain, ...
        xval, yval, xtest, alpha);
    
    %% Evaluation
    ap_val = zeros(size(xval, 3), 1);
    acc_val = zeros(size(xval, 3), 1);
    
    for i = 1:size(xval, 3),
        yhat = yval_pred(:, :, i);
        yc = yval(:, :, i);
        
        [~,~,ap] = compute_ap(yhat(:), yc(:));
        ap_val(i) = ap;
        acc_val(i) = 1-mean(yc(:) ~= (yhat(:) > 0.5));
    end
    
    ap_test = zeros(size(xtest, 3), 1);
    acc_test = zeros(size(xtest, 3), 1);
    
    for i = 1:size(xtest, 3),
        yhat = ytest_pred(:, :, i);
        yc = ytest(:, :, i);
        
        [~,~,ap] = compute_ap(yhat(:), yc(:));
        ap_test(i) = ap;
        acc_test(i) = 1-mean(yc(:) ~= (yhat(:) > 0.5));
    end
    
    %% Display
    fprintf('AP: val = %g (std %g), test = %g (std %g) (alpha = %g) (inference time per image = %g, %s)\n', ...
        mean(ap_val), std(ap_val), mean(ap_test), std(ap_test), alpha, inf_time, paramsname);
    fprintf('ACC: val = %g (std %g), test = %g (std %g) (alpha = %g) (inference time per image = %g, %s)\n', ...
        mean(acc_val), std(acc_val), mean(acc_test), std(acc_test), alpha, inf_time, paramsname);
    
    if (flag_save),
        fid = fopen(sprintf('log/sep_seg_%s.txt', paramsname), 'a+');
        fprintf(fid, '%s\n', paramsname);
        fprintf(fid, 'AP: val = %g (std %g), test = %g (std %g) (alpha = %g) (inference time per image = %g, %s)\n', ...
            mean(ap_val), std(ap_val), mean(ap_test), std(ap_test), alpha, inf_time, paramsname);
        fprintf(fid, 'ACC: val = %g (std %g), test = %g (std %g) (alpha = %g) (inference time per image = %g, %s)\n', ...
            mean(acc_val), std(acc_val), mean(acc_test), std(acc_test), alpha, inf_time, paramsname);
        fprintf(fid, '\n');
        fclose(fid);
    end
    
    %% Visualization
    if flag_visualize,
        mkdir('vis/cnn');
        mkdir(['vis/cnn/' paramsname '/']);
        
        for i = 1:size(xtest, 3),
            xc = xtest(:,:,i);
            yhat = ytest_pred(:,:,i);
            yc = ytest(:,:,i);
            [~,~,ap] = compute_ap(yhat(:),yc(:));
            
            fig = figure(1);
            set(fig, 'Position', [100, 100, 1200, 220]);
            subplot(1,3,1); imagesc(xc); colormap gray; axis off;
            title('original image');
            subplot(1,3,2); imagesc(yc); colormap gray; axis off;
            title('ground truth label');
            subplot(1,3,3); imagesc(yhat); colormap gray; axis off;
            title(sprintf('predicted label, mean AP = %g, studyid = %d', ap, i));
            pause(0.2);
            if (flag_save),
                print(sprintf('%s/%s_prediction_studyid_%d_al_g.png',['vis/cnn/' paramsname '/'], paramsname, i, alpha), fig, '-dpng');
            end
            close(fig);
        end
    end
end

end
