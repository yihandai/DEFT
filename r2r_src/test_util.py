import numpy as np


def compute_fidelity_score(
    critical_frames_starts,
    critical_frames_ends,
    iteration_ends,
    replay_rewards,
    original_rewards,
):
    p_ls = []
    p_ds = []
    batch_size = len(critical_frames_starts)
    for i in range(batch_size):
        critical_frames_start = critical_frames_starts[i]
        critical_frames_end = critical_frames_ends[i]
        iteration_end = iteration_ends[i]

        random_replacement_steps = critical_frames_end - critical_frames_start + 1
        print("r", random_replacement_steps)

        # p_l = random_replacement_steps / iteration_ends
        p_l = random_replacement_steps / iteration_end

        tmp = abs(replay_rewards[i] - original_rewards[i]) / 1
        p_d = tmp if tmp > 0 else 0.001

        p_ls.append(p_l)
        p_ds.append(p_d)
    # print("p_ds", np.log(np.mean(p_ds)))
    # print("p_ls", np.log(np.mean(p_ls)))
    # fidelity_score = -np.log(np.mean(p_ls)) + np.log(np.mean(p_ds))
    print("p_ds", np.mean(np.log(p_ds)))
    print("p_ls", np.mean(np.log(p_ls)))
    fidelity_score = -np.mean(np.log(p_ls)) + np.mean(np.log(p_ds))

    return fidelity_score


def select_critical_steps(mask_probs, iteration_ends, random_zone=False):
    critical_steps_starts = []
    critical_steps_ends = []
    batch_size = len(mask_probs)
    for i in range(batch_size):
        mask_prob = mask_probs[i]
        mask_prob = np.stack(mask_prob)
        iteration_end = iteration_ends[i]

        k = int(iteration_end * 0.2)
        # k = int(iteration_end * 0.3)
        k = max(2, k)

        # k = int(iteration_end * 0.5)
        # k = min(max(5, k), iteration_end)

        if random_zone:
            # random selection instead of top-k confs
            # idx = np.random.choice(iteration_end, size=k, replace=False)
            start = np.random.randint(0, iteration_end - k + 1)
            idx_unsorted = np.arange(start, start + k)
        else:
            # print(mask_prob)
            confs = mask_prob[:, 1]
            # Ensure k doesn't exceed the length of confs
            k = min(k, len(confs))
            # Ensure k is at least 1 for argpartition
            # if k > 0:
            assert k > 0
            # find the top k:
            idx_unsorted = np.argpartition(confs, -k)[-k:]  # Indices not sorted

        # sorted_idxs = idx[np.argsort(confs[idx])][
        #     ::-1
        # ]  # Indices sorted by value from largest to smallest
        idx = idx_unsorted.copy()
        idx.sort()

        critical_steps_start = idx[0]
        critical_steps_end = idx[0]

        ans = 0
        count = 0

        tmp_end = idx[0]
        tmp_start = idx[0]

        for i in range(1, len(idx)):

            # Check if the current element is
            # equal to previous element +1
            if idx[i] == idx[i - 1] + 1:
                count += 1
                tmp_end = idx[i]

            # Reset the count
            else:
                count = 0
                tmp_start = idx[i]
                tmp_end = idx[i]

            # Update the maximum
            if count > ans:
                ans = count
                critical_steps_start = tmp_start
                critical_steps_end = tmp_end
        if critical_steps_end - critical_steps_start == 0:
            critical_steps_start = critical_steps_end = idx_unsorted[0]
        critical_steps_starts.append(critical_steps_start)
        critical_steps_ends.append(critical_steps_end)
    return critical_steps_starts, critical_steps_ends
