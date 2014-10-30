function [weights, grad] = update_params(weights, grad, pos, neg, momentum, epsilon, something_flag)

grad.vishid = momentum .* grad.vishid - epsilon .* bsxfun(@minus, pos.vishid, neg.vishid);
grad.hidvis = momentum .* grad.hidvis - epsilon .* bsxfun(@minus, pos.hidvis, neg.hidvis);
grad.hidbias = momentum .* grad.hidbias - epsilon .* bsxfun(@minus, pos.hidbias, neg.hidbias);
grad.visbias = momentum .* grad.visbias - epsilon .* bsxfun(@minus, pos.visbias, neg.visbias);

weights.vishid = weights.vishid - grad.vishid;
weights.hidvis = weights.hidvis - grad.hidvis;
weights.hidbias = weights.hidbias - grad.hidbias;
weights.visbias = weights.visbias - grad.visbias;

end