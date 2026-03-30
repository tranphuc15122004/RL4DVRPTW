import torch
import torch.nn.functional as F

from itertools import repeat


def _apply_reduction(value, reduction):
    if reduction == 'none':
        return value
    if reduction == 'sum':
        return value.sum()
    return value.mean()


def reinforce_loss(logprobs, rewards, baseline = None, weights = None, discount = 1.0, reduction = 'mean',
                   adv_norm = False, entropy_coef = 0.0, return_components = False):
    r"""
    :param logprobs:     Iterable of length :math:`L` on tensors of size :math:`N \times 1`
    :param rewards:      Iterable of length :math:`L` on tensors of size :math:`N \times 1`
                         or single tensor of size :math:`N \times 1` to use rewards cumulated on the whole trajectory
    :param baseline:     Iterable of length :math:`L` on tensors of size :math:`N \times 1`
                         or single tensor of size :math:`N \times 1` to use rewards cumulated on the whole trajectory
    :param weights:      Iterable of length :math:`L` on tensors of size :math:`N \times 1`
    :param discount:     Discount applied to cumulated future reward
    :param reduction:    'none' No reduction,
                         'sum'  Compute sum of loss on batch,
                         'mean' Compute mean of loss on batch
    :param adv_norm:     If True, normalize advantages to zero mean / unit variance across
                         all timesteps and the entire batch (reduces gradient variance for AC).
    :param entropy_coef: Coefficient for entropy bonus.  A positive value encourages
                         exploration and prevents premature policy collapse.  The logprob
                         of each chosen action is used as a proxy: bonus = entropy_coef * logp
                         (since logp < 0, subtracting it from the loss increases entropy).
    :param return_components:
                         If True, also return a dict with the reduced policy, critic,
                         and entropy contributions that sum to the final loss.
    """
    if weights is None:
        weights = repeat(1.0)

    if isinstance(rewards, torch.Tensor):
        if baseline is None:
            baseline = torch.zeros_like(rewards)

        adv = rewards - baseline.detach()
        if adv_norm:
            adv = (adv - adv.mean()) / (adv.std(unbiased = False) + 1e-8)

        policy_loss = torch.stack([-logp * w for logp, w in zip(logprobs, weights)]).sum(dim = 0)
        policy_loss = policy_loss * adv

        critic_loss = torch.zeros_like(policy_loss)
        if baseline.requires_grad:
            critic_loss = F.smooth_l1_loss(baseline, rewards, reduction = 'none')

        entropy_loss = torch.zeros_like(policy_loss)
        if entropy_coef > 0.0:
            # Entropy bonus: logp < 0, so adding entropy_coef * logp *reduces* certainty
            entropy_loss = torch.stack(logprobs).sum(dim = 0) * entropy_coef

        loss = policy_loss + critic_loss + entropy_loss

    else:
        cumul = torch.zeros_like(rewards[0])
        vals = []
        for r in reversed(rewards):
            cumul = r + discount * cumul
            vals.append(cumul)
        vals.reverse()

        if baseline is None:
            bl_list = [torch.zeros_like(rewards[0]) for _ in vals]
        else:
            bl_list = list(baseline)
            if len(bl_list) < len(vals):
                bl_list.extend(torch.zeros_like(rewards[0]) for _ in range(len(vals) - len(bl_list)))

        # Compute raw advantages for all steps (needed for optional batch-level normalization)
        advs = []
        for val, bl in zip(vals, bl_list):
            advs.append(val - bl.detach())

        if adv_norm and advs:
            all_advs = torch.cat([a.reshape(-1) for a in advs])
            adv_mean = all_advs.mean()
            adv_std  = all_advs.std(unbiased = False) + 1e-8
            advs = [(a - adv_mean) / adv_std for a in advs]

        policy_loss = []
        critic_loss = []
        for val, logp, bl, w, adv in zip(vals, logprobs, bl_list, weights, advs):
            policy_loss.append( -logp * adv * w )
            if bl.requires_grad:
                critic_loss.append(F.smooth_l1_loss(bl, val, reduction = 'none'))
        policy_loss = torch.stack(policy_loss).sum(dim = 0)

        if critic_loss:
            # Keep critic gradients stable across longer trajectories.
            critic_loss = torch.stack(critic_loss).mean(dim = 0)
        else:
            critic_loss = torch.zeros_like(policy_loss)

        entropy_loss = torch.zeros_like(policy_loss)
        if entropy_coef > 0.0:
            entropy_loss = torch.stack(logprobs).sum(dim = 0) * entropy_coef

        loss = policy_loss + critic_loss + entropy_loss

    loss = _apply_reduction(loss, reduction)
    if not return_components:
        return loss

    components = {
        'policy_loss': _apply_reduction(policy_loss, reduction),
        'critic_loss': _apply_reduction(critic_loss, reduction),
        'entropy_loss': _apply_reduction(entropy_loss, reduction),
    }
    return loss, components
