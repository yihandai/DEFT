def average_drop2(
    self,
    img,  # [V, H, W, C] where V = VIEWPOINT_SIZE
    mask_rank,  # [valid_pano, H, W]
    mask,  # [valid_pano, H, W]
    params,
    cls_idx=None,
    verbose=0,
    save_to=None,
    mode="del",
    mask_perc=None,
    topK=None,
    candidate_idx=None,
    causal_metric_dir=None,
):
    # `img` is V images (V = VIEWPOINT_SIZE), and mask has the len of len(candidata)
    assert mode in ["del", "ins"]

    if mode == "ins":
        self.substrate_fn = np.zeros_like

    elif mode == "del":
        # Function that blurs input image
        # Make sure the input dimension is (H, W, C), and GaussianBlur expects (H, W, C)
        # so blur will take a single image array of (H, W, C)
        blur = lambda x: cv2.GaussianBlur(x, (51, 51), 50.0)
        # Optionally, you may want to check dimensions here:
        # if x.ndim != 3 or x.shape[2] != 3:
        #     raise ValueError("Input to blur should be (H, W, C)")
        self.substrate_fn = blur

    NUM_PANO, H, W, C = img.shape

    if type(mask_rank) == torch.Tensor:
        mask_rank = mask_rank.detach().cpu().numpy()  # torch.Tensor --> np.ndarray
        mask = mask.detach().cpu().numpy()  # torch.Tensor --> np.ndarray
    elif type(mask_rank) == np.ndarray:
        mask_rank = mask_rank.copy()
        mask = mask.copy()
    else:
        raise ValueError(f"Invalid mask_rank type: {type(mask_rank)}")

    # upsample images and mask if needed
    if H != self.H or W != self.W:
        img = self.upsample_numpy(img, new_H=self.H, new_W=self.W)
        mask_rank = self.upsample_numpy(mask_rank, new_H=self.H, new_W=self.W)
        mask = self.upsample_numpy(mask, new_H=self.H, new_W=self.W)
    # num_candidate = len(img)
    img_candidate = img[candidate_idx]
    num_candidate = len(img_candidate)

    if cls_idx is None:
        cls_idx = self.call_fn(*params)

    # num_pixels = int(mask_perc * VIEWPOINT_SIZE * HW)
    # only count the pixels in the first batch
    # num_pixels = int(mask_perc * len(candidate_idx) * self.H * self.W)

    if mode == "ins":
        start = []
        for img_i in img_candidate:
            start.append(self.substrate_fn(img_i))
        start = np.stack(start)
        finish = img_candidate.copy()

    elif mode == "del":
        start = img_candidate.copy()
        finish = []
        for img_i in img_candidate:
            finish.append(self.substrate_fn(img_i))
        finish = np.stack(finish)

    # 输入形状 (V, H, W) where V = VIEWPOINT_SIZE
    # 输入形状 (V, H, W) where V = VIEWPOINT_SIZE
    if mask_perc is not None:
        topK = int(mask_rank.max() * mask_perc)
    flat_mask_rank = mask_rank.reshape(-1)  # 展平
    flat_mask = mask.reshape(-1)  # 展平
    valid_idx = np.where((flat_mask_rank >= 0) & (flat_mask_rank < topK + 1))[
        0
    ]  # 过滤值在 [1,6) 范围的像素索引
    non_valid_idx = np.where((flat_mask_rank < 0) | (flat_mask_rank >= topK + 1))[0]

    # 只在这些索引里做排序
    coords = valid_idx[np.argsort(flat_mask_rank[valid_idx])[::-1]]
    # salient_order 依然是一维索引，代表满足条件且排序后的像素位置

    # salient_order = np.argsort(mask.reshape(-1))[::-1]  # 输出形状 (B*H*W,)
    # print(salient_order.shape)

    # coords = salient_order[:num_pixels]

    # 1. [B, H, W, 3] --> [B, 3, W, H]
    # 2. 展平为 (B, 3, HW)，HW = H * W
    start_flat = start.transpose(0, 3, 1, 2).reshape(
        num_candidate, 3, -1
    )  # -1 自动计算HW
    finish_flat = finish.transpose(0, 3, 1, 2).reshape(num_candidate, 3, -1)

    # 将全局坐标 coords 分解为子图编号和子图内坐标
    subimage_ids, pixel_indices = (
        coords // (self.H * self.W),
        coords % (self.H * self.W),
    )

    # 批量赋值（无需循环，直接向量化操作）
    start_flat[subimage_ids, :, pixel_indices] = finish_flat[
        subimage_ids, :, pixel_indices
    ]

    prediction = self.call_fn(*params, new_imgs=[start], candidata_list=[candidate_idx])

    if self.target == "MapGPT":
        print("gt{}\tprediction{}".format(cls_idx, prediction[0]))
        cls_idx_new = prediction[0]

    if mode == "ins":
        self.insertion_curr[mask_perc]["num"] += 1
        if cls_idx_new[0] == cls_idx:
            curr = 1
        else:
            curr = 0
        self.insertion_curr[mask_perc]["curr"] += curr
        self.insertion_curr[mask_perc]["rate"] = (
            self.insertion_curr[mask_perc]["curr"]
            / self.insertion_curr[mask_perc]["num"]
        )
        print(
            "Insertion sample: {}. Over-all: {} for mask percentage: {}".format(
                curr, self.insertion_curr[mask_perc]["rate"], mask_perc
            )
        )
    else:
        self.deletion_curr[mask_perc]["num"] += 1
        if cls_idx_new[0] == cls_idx:
            curr = 1
        else:
            curr = 0
        self.deletion_curr[mask_perc]["curr"] += curr
        self.deletion_curr[mask_perc]["rate"] = (
            self.deletion_curr[mask_perc]["curr"] / self.deletion_curr[mask_perc]["num"]
        )
        print(
            "Deletion sample: {}. Over-all: {} for mask percentage: {}".format(
                curr, self.deletion_curr[mask_perc]["rate"], mask_perc
            )
        )
    # NOTE: ?
    if args.feature_level_baseline == "smdl":
        flat_mask = flat_mask[::-1]
    if mode == "ins":
        self.collect_consistency_importance_score(
            params[1],
            params[2],
            curr,
            flat_mask[non_valid_idx],
            causal_metric_dir,
            mask_perc=mask_perc,
            mode="Insertion",
        )
    else:
        self.collect_consistency_importance_score(
            params[1],
            params[2],
            curr,
            flat_mask[valid_idx],
            causal_metric_dir,
            mask_perc=mask_perc,
            mode="Deletion",
        )
    # save temporary results in a better structured format
    temp_results = {
        "mode": mode,
        "mask_percentage": mask_perc,
        "stats": (
            self.insertion_curr[mask_perc]
            if mode == "ins"
            else self.deletion_curr[mask_perc] if mode == "del" else None
        ),
    }

    causal_metric_dir = os.path.join("snap", args.name, "causal_metric")
    if temp_results["stats"] is None:
        raise ValueError(f"Invalid mode: {mode}")
