addpath cnn/;
addpath utils/;
%addpath filters/;
if ~exist('results', 'dir'),
    mkdir('results');
end
if ~exist('log', 'dir'),
    mkdir('log');
end
if ~exist('vis', 'dir'),
    mkdir('vis');
	mkdir('vis/cnn');
    mkdir('vis/hidden_units');
end

if ~exist('original_data/','dir')
	mkdir('original_data/');
end

if ~exist('normalized_data/','dir');
	mkdir('normalized_data/','dir');
end
