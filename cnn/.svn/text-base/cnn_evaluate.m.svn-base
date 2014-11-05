function [rec, prec, ap] = cnn_evaluate(x, y, weights, params)

h = cnn_infer(x, weights, params);
yhat = cnn_recon(h, weights, params);

[rec, prec, ap] = compute_ap(yhat(:), y(:));

return;