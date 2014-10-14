function cost = cross_entropy(x, xr)

x = x(:);
xr = xr(:);

% for numerical stability
xr = min(xr, 1-1e-8);
xr = max(xr, 1e-8);

cost = -sum(x.*log(xr)+(1-x).*log(1-xr));

return;

