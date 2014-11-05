function weights = cnn_unroll(theta, params)

idx = 0;

weights.vishid = reshape(theta(idx+1:idx+params.ws^2*params.numch*params.numhid), ...
    params.ws, params.ws, params.numch, params.numhid);
idx = idx + numel(weights.vishid);

weights.hidbias = theta(idx+1:idx+params.numhid);
idx = idx + numel(weights.hidbias);

weights.hidvis = reshape(theta(idx+1:idx+params.ws^2*params.numhid*params.numout), ...
    params.ws, params.ws, params.numhid, params.numout);
idx = idx + numel(weights.hidvis);

weights.visbias = reshape(theta(idx+1:idx+params.rs*params.cs*params.numout), params.rs, params.cs, params.numout);
idx = idx + numel(weights.visbias);

assert(idx == length(theta));


return;