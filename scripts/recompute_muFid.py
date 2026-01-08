import os
import numpy as np


def read_importace_consistency_score(causal_metric_dir, dir_):
    importance_score = []
    consistency_score = []
    consistency_dict = {
        "ins": {
            "0.25": {"sum": 0, "num": 0, "ratio": 0},
            "0.5": {"sum": 0, "num": 0, "ratio": 0},
            "0.75": {"sum": 0, "num": 0, "ratio": 0},
        },
        "del": {
            "0.25": {"sum": 0, "num": 0, "ratio": 0},
            "0.5": {"sum": 0, "num": 0, "ratio": 0},
            "0.75": {"sum": 0, "num": 0, "ratio": 0},
        },
    }
    for instr_id in os.listdir(causal_metric_dir):
        for t in os.listdir(os.path.join(causal_metric_dir, instr_id)):
            for mask_perc in os.listdir(os.path.join(causal_metric_dir, instr_id, t)):
                # for mode in ["ins", "del"]:
                for mode in ["ins"]:
                    # for mode in ["ins"]:
                    # if mode == "del" and (mask_perc == "0.5" or mask_perc == "0.25"):
                    if mode == "del" and (mask_perc == "0.5" or mask_perc == "0.75"):
                        # dir_ = causal_metric_dir
                        # dir_ = "./snap/VLNBERT-test-navgpt2-ensemblev1_ensemble"
                        # dir_ = "./snap/VLNBERT-test-baseline-navgpt2-smdlv1"
                        # suffix_ = "causal_metric_pixel_update_replication_1/consistency_importance_score"
                        suffix_ = "causal_metric_pixel_2/consistency_importance_score"
                        causal_metric_dir_ = os.path.join(dir_, suffix_)
                        score = np.load(
                            os.path.join(
                                causal_metric_dir_,
                                instr_id,
                                t,
                                mask_perc,
                                mode,
                                "score.npy",
                            )
                        )
                    else:
                        score = np.load(
                            os.path.join(
                                causal_metric_dir,
                                instr_id,
                                t,
                                mask_perc,
                                mode,
                                "score.npy",
                            )
                        )
                    # importance_score.append(score[1])
                    importance_score.append(recompute_importace(mask_perc, mode))
                    consistency_score.append(score[0])
                    consistency_dict[mode][mask_perc]["sum"] += score[0]
                    consistency_dict[mode][mask_perc]["num"] += 1
                    consistency_dict[mode][mask_perc]["ratio"] = (
                        consistency_dict[mode][mask_perc]["sum"]
                        / consistency_dict[mode][mask_perc]["num"]
                    )
    print("consistency_dict", consistency_dict)
    return importance_score, consistency_score


def compute_muFidelity(importance_score, consistency_score):
    consistency_arr = np.asarray(consistency_score, dtype=np.float64)
    importance_arr = np.asarray(importance_score, dtype=np.float64)

    consistency_impact = 1.0 - consistency_arr

    corr_matrix = np.corrcoef(importance_arr, consistency_impact)
    corr = corr_matrix[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def recompute_importace(mask_perc, mode):
    mask_perc = float(mask_perc)
    H, W = 480, 640
    # assign importance score to each pixel
    pixel_num = H * W
    pixel_importance_list = np.arange(pixel_num)[::-1]
    if mode == "ins":
        return pixel_importance_list[int(mask_perc * pixel_num) :].sum()
    else:
        return pixel_importance_list[: int(mask_perc * pixel_num)].sum()


if __name__ == "__main__":
    # # dir_list = [
    # #     "./snap/VLNBERT-test-baseline-mapgpt-igv3",
    # #     "./snap/VLNBERT-train-feature-mapgpt-ensemblev3",
    # #     "./snap/VLNBERT-test-baseline-mapgpt-random-randomv3",
    # #     # "./snap/VLNBERT-test-baseline-mapgpt-fg_camv2",
    # #     "./snap/VLNBERT-test-baseline-mapgpt-smdlv3",
    # # ]
    # # dir_list = [
    # #     "./snap/VLNBERT-test-navgpt2-ensemblev1_ensemble",
    # #     "./snap/VLNBERT-test-baseline-navgpt2-guided-igv1",
    # #     "./snap/VLNBERT-test-baseline-navgpt2-igv1",
    # #     "./snap/VLNBERT-test-baseline-navgpt2-smdlv1",
    # #     "./snap/VLNBERT-test-baseline-navgpt2-hsicv1",
    # # ]
    # dir_list = [
    #     # "./snap/VLNBERT-test-navgpt-ensemblev3",
    #     "./snap/VLNBERT-test-baseline-navgpt-guided-igv1",
    #     "./snap/VLNBERT-test-baseline-navgpt-igv1",
    #     "./snap/VLNBERT-test-baseline-navgpt-smdlv1",
    #     # "./snap/VLNBERT-test-baseline-navgpt2-hsicv1",
    # ]
    # suffix_update = ""
    # suffix = "causal_metric_pixel" + suffix_update + "/consistency_importance_score/"
    # causal_metric_dir_list = [os.path.join(dir, suffix) for dir in dir_list]
    # for i, causal_metric_dir in enumerate(causal_metric_dir_list):
    #     print(dir_list[i])
    #     importance_score, consistency_score = read_importace_consistency_score(
    #         causal_metric_dir, dir_list[i]
    #     )
    #     # print("importance_score", importance_score)
    #     # print("consistency_score", consistency_score)
    #     muFidelity = compute_muFidelity(importance_score, consistency_score)
    #     print("muFidelity", muFidelity)

    # dir_replication = "./snap/VLNBERT-test-baseline-mapgpt-smdlv3"
    # dir_replication = "./snap/VLNBERT-test-navgpt2-ensemblev1_ensemble"

    # dir_replication = "./snap/VLNBERT-test-navgpt-ensemblev3"
    suffix = "causal_metric_pixel_2/consistency_importance_score"
    dir_replication = "./snap/VLNBERT-test-baseline-navgpt-smdlv1"
    # dir_replication = "./snap/VLNBERT-test-baseline-navgpt2-smdlv1"
    # dir_replication = "./snap/VLNBERT-test-baseline-navgpt2-igv1"
    causal_metric_dir_replication = os.path.join(dir_replication, suffix)
    print(causal_metric_dir_replication)
    importance_score, consistency_score = read_importace_consistency_score(
        causal_metric_dir_replication, dir_replication
    )
    muFidelity = compute_muFidelity(importance_score, consistency_score)
    print("muFidelity", muFidelity)
