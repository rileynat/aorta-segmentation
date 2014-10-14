function theta = cnn_roll(weights)

theta = [];
theta = [theta ; weights.vishid(:)];
theta = [theta ; weights.hidbias(:)];
theta = [theta ; weights.hidvis(:)];
theta = [theta ; weights.visbias(:)];

return;