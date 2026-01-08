import torch
import numpy as np
from r2r_src.gail_utils import get_entropy, log_prob_density
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_discrim(discrim, memory, discrim_optim, demonstrations, args, logs):
    # memory
    hidden_states = torch.cat([item[0].detach() for item in memory]).to(
        device
    )  # stack state tensors
    actions = torch.cat([item[1].detach() for item in memory]).to(
        device
    )  # stack action tensors
    action_feats = torch.cat([item[5].detach() for item in memory]).to(device)
    masks = np.concatenate([item[3] for item in memory])
    valid_mask = masks > 0

    # demonstrations
    hidden_states_d = torch.cat([item[0].detach() for item in demonstrations]).to(
        device
    )
    actions_d = torch.cat([item[1].detach() for item in demonstrations]).to(device)
    action_feats_d = torch.cat([item[5].detach() for item in demonstrations]).to(device)
    masks_d = np.concatenate([item[3] for item in demonstrations])
    valid_mask_d = masks_d > 0

    # filter invalid transitions
    hidden_states = hidden_states[valid_mask]
    actions = actions[valid_mask]
    action_feats = action_feats[valid_mask]

    hidden_states_d = hidden_states_d[valid_mask_d]
    actions_d = actions_d[valid_mask_d]
    action_feats_d = action_feats_d[valid_mask_d]

    criterion = torch.nn.BCELoss()

    if actions.dim() == 1:
        actions = actions.unsqueeze(1)
    if actions_d.dim() == 1:
        actions_d = actions_d.unsqueeze(1)

    for _ in range(args.discrim_update_num):
        # learner = torch.cat([hidden_states, actions], dim=1)
        learner = torch.cat([hidden_states, action_feats], dim=1)
        learner_output = discrim(learner).unsqueeze(1)
        # expert = torch.cat(
        #     [hidden_states_d, actions_d],
        #     dim=1,
        # )
        expert = torch.cat(
            [hidden_states_d, action_feats_d],
            dim=1,
        )
        expert_output = discrim(expert).unsqueeze(1)

        learner_target = torch.ones((hidden_states.shape[0], 1), device=device)
        expert_target = torch.zeros((hidden_states_d.shape[0], 1), device=device)

        discrim_loss = criterion(learner_output, learner_target) + criterion(
            expert_output, expert_target
        )
        logs["DISC_LOSS"].append(discrim_loss.item())

        discrim_optim.zero_grad()
        discrim_loss.backward()
        discrim_optim.step()
    # expert_acc = (
    #     (discrim(torch.cat([hidden_states_d, actions_d], dim=1)) < 0.5).float()
    # ).mean()
    # learner_acc = (
    #     (discrim(torch.cat([hidden_states, actions], dim=1)) > 0.5).float()
    # ).mean()

    expert_acc = (
        (discrim(torch.cat([hidden_states_d, action_feats_d], dim=1)) < 0.5).float()
    ).mean()
    learner_acc = (
        (discrim(torch.cat([hidden_states, action_feats], dim=1)) > 0.5).float()
    ).mean()

    return expert_acc, learner_acc


def train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args, logs):

    hidden_states = torch.cat(
        [item[0].detach() for item in memory]
    )  # stack state tensors
    actions = torch.cat([item[1].detach() for item in memory])  # stack action tensors

    # rewards = list(memory[:, 2])
    rewards = np.concatenate([x[2].detach().cpu() for x in memory])
    # masks = list(memory[:, 3])
    masks = np.concatenate([x[3] for x in memory])

    def pad_and_cat(tensor_list, dim=1, pad_value=0):
        """
        Pad a list of tensors along `dim` to the maximum length and then concatenate along batch (dim=0).
        """
        max_len = max(t.shape[dim] for t in tensor_list)
        padded_list = []
        for t in tensor_list:
            pad_size = [0, 0] * (t.dim() - dim - 1) + [0, max_len - t.shape[dim]]
            padded_list.append(F.pad(t, pad_size, value=pad_value))
        return torch.cat(padded_list, dim=0).detach()

    states = {
        "mode": "visual",
        "sentence": pad_and_cat([x[4]["sentence"] for x in memory]),
        "attention_mask": pad_and_cat([x[4]["attention_mask"] for x in memory]),
        "lang_mask": pad_and_cat([x[4]["lang_mask"] for x in memory]),
        "vis_mask": pad_and_cat([x[4]["vis_mask"] for x in memory]),
        "token_type_ids": pad_and_cat([x[4]["token_type_ids"] for x in memory]),
        "action_feats": pad_and_cat([x[4]["action_feats"] for x in memory]),
        "cand_feats": pad_and_cat([x[4]["cand_feats"] for x in memory]),
    }

    # print("hidden_states shape:", hidden_states.shape)
    n = hidden_states.shape[0]
    arr = np.arange(n)

    old_values = critic(hidden_states).detach()

    returns, advants = get_gae(rewards, masks, old_values, args)

    old_policy = []
    for idx in range(n // args.batchSize):
        idx_start = idx * args.batchSize
        idx_end = min(idx_start + args.batchSize, n)
        idxs = arr[idx_start:idx_end]
        states_tmp = {
            "mode": "visual",
            "sentence": states["sentence"][idxs],
            "attention_mask": states["attention_mask"][idxs],
            "lang_mask": states["lang_mask"][idxs],
            "vis_mask": states["vis_mask"][idxs],
            "token_type_ids": states["token_type_ids"][idxs],
            "action_feats": states["action_feats"][idxs],
            "cand_feats": states["cand_feats"][idxs],
        }
        h_t, logit = actor(**states_tmp)
        probs = F.log_softmax(logit, 1)
        c = torch.distributions.Categorical(probs)
        old_policy.append(c.log_prob(actions[idxs]))
    old_policy = torch.cat(old_policy).detach()

    criterion = torch.nn.MSELoss()

    for _ in range(args.actor_critic_update_num):
        np.random.shuffle(arr)
        for i in range(n // args.batchSize):
            batch_index = arr[args.batchSize * i : args.batchSize * (i + 1)]
            batch_index = torch.LongTensor(batch_index)

            hidden_states_samples = hidden_states[batch_index]

            values = critic(hidden_states_samples)

            actions_samples = actions[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()

            clipped_values = oldvalue_samples + torch.clamp(
                values - oldvalue_samples, -args.clip_param, args.clip_param
            )
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            inputs = {
                "mode": "visual",
                "sentence": states["sentence"][batch_index],
                "attention_mask": states["attention_mask"][batch_index],
                "lang_mask": states["lang_mask"][batch_index],
                "vis_mask": states["vis_mask"][batch_index],
                "token_type_ids": states["token_type_ids"][batch_index],
                "action_feats": states["action_feats"][batch_index],
                "cand_feats": states["cand_feats"][batch_index],
            }
            loss, ratio, entropy = surrogate_loss(
                actor,
                advants_samples,
                inputs,
                old_policy.detach(),
                actions_samples,
                batch_index,
            )

            clipped_ratio = torch.clamp(
                ratio, 1.0 - args.clip_param, 1.0 + args.clip_param
            )
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss_total = actor_loss + 0.5 * critic_loss - 0.001 * entropy
            logs["AC_LOSS"].append(loss_total.item())
            critic_optim.zero_grad()
            actor_optim.zero_grad()
            loss_total.backward()
            critic_optim.step()
            actor_optim.step()


def train_actor_critic_v2(actor, critic, memory, actor_optim, critic_optim, args, logs):
    max_time = len(memory)
    hidden_states = torch.stack(
        [item[0].detach() for item in memory]
    )  # stack state tensors
    actions = torch.stack([item[1].detach() for item in memory])  # stack action tensors

    # rewards = list(memory[:, 2])
    rewards = np.stack([x[2].detach().cpu() for x in memory])
    # masks = list(memory[:, 3])
    masks = np.stack([x[3] for x in memory])
    # states = [x[4] for x in memory]
    states = []
    for i, mem in enumerate(memory):
        state_ori = mem[4]
        state_tmp = dict()
        for k, v in state_ori.items():
            if torch.is_tensor(v):
                state_tmp[k] = v.detach()
            else:
                state_tmp[k] = v
        states.append(state_tmp)

    # old_values = critic(hidden_states).detach()
    old_values = []
    old_policy = []
    for t in range(max_time):
        old_values.append(critic(hidden_states[t]))

        h_t, logit = actor(**(states[t]))
        probs = F.log_softmax(logit, 1)
        c = torch.distributions.Categorical(probs)
        old_policy.append(c.log_prob(actions[t]))
    old_values = torch.stack(old_values).detach()
    old_policy = torch.stack(old_policy).detach()

    returns, advants = get_gae(rewards, masks, old_values, args)

    criterion = torch.nn.MSELoss()

    for _ in range(args.actor_critic_update_num):
        # np.random.shuffle(arr)
        t = torch.randint(0, max_time, (1,))[0]

        hidden_states_samples = hidden_states[t]

        values = critic(hidden_states_samples)

        actions_samples = actions[t]
        # returns_samples = returns.unsqueeze(1)[t]
        # advants_samples = advants.unsqueeze(1)[t]
        returns_samples = returns[t]
        advants_samples = advants[t]
        oldvalue_samples = old_values[t].detach()
        mask_t = torch.tensor(masks[t]).to(values.device)

        clipped_values = oldvalue_samples + torch.clamp(
            values - oldvalue_samples, -args.clip_param, args.clip_param
        )
        critic_loss1 = criterion(clipped_values, returns_samples)
        critic_loss2 = criterion(values, returns_samples)
        # critic_loss = torch.max(critic_loss1, critic_loss2).mean()
        critic_loss = (torch.max(critic_loss1, critic_loss2) * mask_t).sum() / (
            mask_t.sum() + 1e-5
        )

        inputs = states[t]
        loss, ratio, entropy = surrogate_loss(
            actor,
            advants_samples,
            inputs,
            old_policy.detach(),
            actions_samples,
            t,
        )

        clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param)
        clipped_loss = clipped_ratio * advants_samples
        # actor_loss = -torch.min(loss, clipped_loss).mean()
        actor_loss = -torch.min(loss, clipped_loss) * mask_t
        actor_loss = actor_loss.sum() / (mask_t.sum() + 1e-5)

        loss_total = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        logs["AC_LOSS"].append(loss_total.item())
        critic_optim.zero_grad()
        actor_optim.zero_grad()
        loss_total.backward()
        critic_optim.step()
        actor_optim.step()


def get_gae(rewards, masks, values, args):
    # rewards = torch.Tensor(rewards)
    # masks = torch.Tensor(masks)
    rewards = torch.from_numpy(rewards).to(device)
    masks = torch.from_numpy(masks).to(device)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (args.gamma * running_returns * masks[t])
        returns[t] = running_returns

        running_delta = (
            rewards[t] + (args.gamma * previous_value * masks[t]) - values.data[t]
        )
        previous_value = values.data[t]

        running_advants = running_delta + (
            args.gamma * args.lamda * running_advants * masks[t]
        )
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    # TODO: probs
    h_t, logit = actor(**states)
    probs = F.log_softmax(logit, 1)
    c = torch.distributions.Categorical(probs)
    new_policy = c.log_prob(actions)

    # mu, std = actor(**states)
    # new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    # TODO: entropy
    entropy = c.entropy().mean()

    return surrogate_loss, ratio, entropy
