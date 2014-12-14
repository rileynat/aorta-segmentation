function [] = test_cnn_framework(test_data, test_labels, weights, data_set)

filterInfo = struct;
filterInfo.numFilters1 = 30;
filterInfo.filterSize1 = 5;
filterInfo.numFilters2 = 20;
filterInfo.filterSize2 = 16;
filterInfo.numFilters3 = filterInfo.numFilters2;
filterInfo.filterSize3 = filterInfo.filterSize1 + filterInfo.filterSize2 - 1;
filterInfo.data_set = data_set;

addpath('cnn');
[ap, acc, test_pred] = test_cnn(test_data, test_labels, weights, filterInfo);

fprintf('AP: %g\n', mean(ap));

info = clock;
mat_name = sprintf('results/%d-%d-%d-%d:%d-set-%d', info(1), info(2), info(3), info(4), info(5), data_set);
save(mat_name, 'ap', 'acc', 'test_pred');

end
