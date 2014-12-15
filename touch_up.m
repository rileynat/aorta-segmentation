function [AP] = touch_up(data, labels, pred)
    
    addpath('utils');
    AP = size(data, 3);

    
    for k = 1:size(pred, 3)
        xc = data(:,:,k);
        f = figure('Visible','off','Position',[360,500,450,285]);
        imagesc(xc); colormap gray; axis off;
        
        % Get valid rectangle
        out = getrect(f);
        valid_ind = round(out);
        ind = ones(size(xc));
        ind(valid_ind(2): (valid_ind(2) + valid_ind(4)), valid_ind(1):(valid_ind(1) + valid_ind(3))) = 0;
        ind = logical(ind);
        yhat = pred(:,:,k);
        yhat(ind) = 0;
        yc = labels(:,:,k);
        
        [tc, bc, lc, rc] = find_edges(yhat);

        
        % Display prediction boundaries
        hold on;
        %imagesc(xc); colormap gray; hold on;
        a = 40;
        scatter(tc{1}(:, 1), tc{1}(:, 2), a, 'r.');
        scatter(bc{1}(:, 1), bc{1}(:, 2), a, 'r.');
        scatter(lc{1}(:, 1), lc{1}(:,2), a, 'r.');
        scatter(rc{1}(:,1), rc{1}(:,2), a, 'r.');
        
        cont = 1;
        % "Touch up" until user is finished
        btn = uicontrol('Style', 'togglebutton', 'String', 'DONE', ...
            'Position', [500 20 50 20]);
        mode = uicontrol('Style', 'togglebutton', 'String', 'Include', ...
            'Position', [1000 20 50 20], 'Callback', @mode_callback);
        while cont
           [x, y] = ginput(1);
           if get(btn, 'Value')
               break
           end
           x = round(x);
           y = round(y);
           if y > size(data, 1) || x > size(data,2)
               continue;
           end
           val = data(y, x, k);
           rad = 1;
           sample = data(y-rad:y+rad, x-rad: x+rad, k);
           diffs = sqrt((sample - val).^2)
           ind = diffs < 10;
           updated = yhat(y-rad:y+rad, x-rad:x+rad);
           
           if get(mode, 'Value') == 0
                updated(ind) = updated(ind) + .5;
           end
           
           if get(mode, 'Value') == 1
               updated(ind) = updated(ind) - .5;
               neg_ind = updated < 0;
               updated(neg_ind) = 0;
           end
           
           
           yhat(y-rad:y+rad, x-rad:x+rad) = updated;
           [tc, bc, lc, rc] = find_edges(yhat);
           hold off;
           imagesc(xc); colormap gray; axis off; hold on;
           scatter(tc{1}(:, 1), tc{1}(:, 2), a, 'r.');
           scatter(bc{1}(:, 1), bc{1}(:, 2), a, 'r.');
           scatter(lc{1}(:, 1), lc{1}(:,2), a, 'r.');
           scatter(rc{1}(:,1), rc{1}(:,2), a, 'r.');         
           
        end
        
        
        [~,~,ap] = compute_ap(yhat(:),yc(:));
        AP(k) = ap;
        
        fig = figure(1);
        set(fig, 'Position', [100, 100, 1200, 220]);
        subplot(1,3,1); imagesc(xc); colormap gray; axis off;
        title('original image');
        subplot(1,3,2); imagesc(yc); colormap gray; axis off;
        title('ground truth label');
        subplot(1,3,3); imagesc(yhat); colormap gray; axis off;
        title(sprintf('predicted label, mean AP = %g, studyid = %d', ap, i));
        pause(2);
        
    end


end

function mode_callback(mode,data)
    if get(mode, 'Value') == 1
        set(mode, 'String', 'Exclude');
    end
    if get(mode, 'Value') == 0
        set(mode, 'String', 'Include');
    end
end
