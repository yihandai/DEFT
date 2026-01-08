def compute_random_salency(
    self,
    obs,
    # gmaps,
    t,
    h_t_input,
    language_features,
    language_inputs,
    language_attention_mask,
    token_type_ids,
    # steps=50,
    # mode="IG",
):

    bs = len(obs)
    # get panorama images and transform them -> np
    images_numpys = []
    for i in range(bs):
        scanId = obs[i]["scan"]
        viewpointId = obs[i]["viewpoint"]
        images_numpy = self.get_vp_images(self.sim, scanId, viewpointId)
        images_numpys.append(images_numpy)
    images_numpys = np.stack(images_numpys)  # [bs, vp, C, H, W]

    images_return = np.array(
        [
            [self.reverse_transforms(image) for image in images]
            for images in images_numpys
        ]
    )  # [B, vp, H, W, C] where vp = VIEWPOINT_SIZE
    B, V, C, H, W = images_numpys.shape
    candidata_list = self.get_only_can_list(obs)

    # gen a random salency map [bs, vp, 224, 224]
    # Generate spatially coherent random saliency using random Gaussian blobs
    # This creates more natural-looking clusters compared to simple downsampling
    heatmaps = []
    for i in range(bs):
        heatmap = np.zeros((self.VIEWPOINT_SIZE, 3, H, W), dtype=np.float32)

        for vp in range(self.VIEWPOINT_SIZE):
            # Create coordinate grids once per viewpoint
            y_coords, x_coords = np.mgrid[0:H, 0:W].astype(np.float32)

            # Generate blobs for all 3 channels together
            blob_map_all_channels = np.zeros((3, H, W), dtype=np.float32)

            # Generate random Gaussian blobs (clusters)
            num_blobs = np.random.randint(3, 8)  # Random number of clusters

            for _ in range(num_blobs):
                # Random center position
                center_y = np.random.uniform(0, H)
                center_x = np.random.uniform(0, W)

                # Random blob size (sigma for Gaussian)
                sigma_y = np.random.uniform(H * 0.1, H * 0.3)
                sigma_x = np.random.uniform(W * 0.1, W * 0.3)

                # Random intensity for each of the 3 channels
                intensities = np.random.uniform(50, 255, size=3)

                # Create Gaussian blob (same shape for all channels)
                gaussian = np.exp(
                    -(
                        (x_coords - center_x) ** 2 / (2 * sigma_x**2)
                        + (y_coords - center_y) ** 2 / (2 * sigma_y**2)
                    )
                )

                # Apply different intensities to each channel
                blob_map_all_channels += (
                    gaussian[np.newaxis, :, :] * intensities[:, np.newaxis, np.newaxis]
                )

            # # Normalize each channel to [0, 255] range
            # for c in range(3):
            #     blob_map = blob_map_all_channels[c]
            #     blob_map = (blob_map - blob_map.min()) / (
            #         blob_map.max() - blob_map.min() + 1e-8
            #     )
            #     blob_map = blob_map * 255
            #     heatmap[vp, c] = blob_map
            heatmap[vp] = blob_map_all_channels

        heatmap = self.gen_heatmap(heatmap)
        heatmaps.append(heatmap)

    return images_return, heatmaps, candidata_list
