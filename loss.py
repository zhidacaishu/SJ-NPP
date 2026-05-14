import torch

def loss_time(idx, time, out_time, out, model, sample=1):
    """loss function for time"""
    T = time[-1]
    t = torch.FloatTensor(sample).uniform_(0, T).to(T.device)

    elapsed_t = t.unsqueeze(dim=-1) - time.unsqueeze(dim=0)
    pos_ind = (elapsed_t>0)
    elapsed_t = elapsed_t * pos_ind
    time_decay = torch.exp(-model.sftp(model.w[idx, 0]) * elapsed_t) * pos_ind

    ejc = torch.exp(model.sftp(model.attn_unit(out[1:]))[:, 0].unsqueeze(dim=0) * time_decay) * pos_ind
    sum_ejc = torch.sum(ejc, dim=-1).unsqueeze(dim=-1)
    sum_ejc[sum_ejc==0] = 1

    aw = ejc / sum_ejc
    sn = aw.matmul(out[1:])
    left_out_index = pos_ind.sum(dim=1)

    mc_cif = model.sftp(model.h2o_time(sn + out[left_out_index]) + model.h2o_time_bias[idx])
    
    loss_1 = torch.log(out_time).sum()
    loss_2 = T * torch.mean(mc_cif)
    return -(loss_1 - loss_2)

def loss_action(action, out_action):
    """loss function for action: negative log-likelihood"""

    action_loss = torch.log(torch.gather(out_action, dim=1, index=action.long().unsqueeze(1)).squeeze(1)).sum()

    return -action_loss


def loss_item(item, out_item):
    """loss function for item: negative log-likelihood"""

    item_loss = torch.log(torch.gather(out_item, dim=1, index=item.long().unsqueeze(1)).squeeze(1)).sum()

    return -item_loss
