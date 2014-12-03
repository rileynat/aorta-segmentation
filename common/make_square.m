function [ square_data ] = make_square( data )
% Makes the image square by truncating the image along it's larger
% dimension.

final_dim = min(size(data, 1), size(data, 2));

dim1 = (size(data, 1) - final_dim)/2;
dim1_lower = floor(dim1);
dim1_upper = round(dim1);

dim2 = (size(data, 2) - final_dim)/2;
dim2_lower = floor(dim2);
dim2_upper = round(dim2);

square_data = data(dim1_lower+1:(size(data, 1)-dim1_upper), ...
    dim2_lower+1:(size(data,2)-dim2_upper), :);
end

