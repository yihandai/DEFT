import os
import json
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import numpy as np

def load_result(dir_name):
    result = []
    file_name_list = os.listdir(dir_name)
    for json_file in file_name_list:
        json_path = os.path.join(dir_name, json_file)

        name, SR, spl, unc, gt, pred = read_jsonfile(json_path)
        result.append({"instr_id": name, "SR": SR, "spl": spl, "uncertainty": unc, \
                       "gt": gt, "pred": pred})
    
    return result

def read_jsonfile(file_name):
    with open(file_name, "r") as file:
        json_obj = json.load(file)
        name = json_obj["instr_id"]
        SR = json_obj["evaluation"]["success"]
        spl = json_obj["evaluation"]["spl"]
        unc = json_obj["uncertainty"]
        gt = json_obj["gt_traj"]
        pred = json_obj["trajectory"]
        # ne = json_obj[""]
    return name, SR, spl, unc, gt, pred

def draw_relation_unc_and_SR(data):
    def mean_list(l):
        sum_l = 0
        for item, unc in l.items():
            sum_l += unc
        return sum_l / len(l)

    uncertainties = []
    SRs = []
    for i, traj in enumerate(data):
        uncertainties.append(mean_list(traj["uncertainty"]))
        SRs.append(traj["SR"])
    plt.scatter(uncertainties, SRs)
    plt.savefig("./datasets/exprs_map/test/pic/SR.pdf")

def draw_relation_unc_and_spl(data):
    def mean_list(l):
        sum_l = 0
        for item, unc in l.items():
            sum_l += unc
        return sum_l / len(l)

    uncertainties = []
    spls = []
    for i, traj in enumerate(data):
        uncertainties.append(mean_list(traj["uncertainty"]))
        spls.append(traj["spl"])
    plt.scatter(uncertainties, spls)
    plt.savefig("./datasets/exprs_map/test/pic/spl.pdf")
    # plt.show()

def show_unc_trending(data, success):
    for i, traj in enumerate(data):
        if traj["SR"][0] == success:
            time_sequence = [x/(len(traj["uncertainty"])-1) for x in range(len(traj["uncertainty"]))]
            uncertainty_trending = [x for x in traj["uncertainty"].values()]
            # print(traj["SR"])
            plt.plot(time_sequence, uncertainty_trending)
    # plt.show()
    plt.savefig("./datasets/exprs_map/test/pic/uncertainty_{}.pdf".format(str(success)))

def show_unc_difference(data, success):
    for i, traj in enumerate(data):
        if traj["SR"][0] == success:
            time_sequence = [x/(len(traj["uncertainty"])-2) for x in range(len(traj["uncertainty"])-1)]
            # uncertainty_trending = [x for x in traj["uncertainty"].values()]
            uncertainty_trending = []
            y_t = None
            for y_t_plus_1 in traj["uncertainty"].values():
                if y_t == None:
                    y_t = y_t_plus_1
                else:
                    uncertainty_trending.append(y_t_plus_1 - y_t)
                    y_t = y_t_plus_1
            # print(traj["SR"])
            plt.plot(time_sequence, uncertainty_trending)
    # plt.show()
    plt.savefig("./datasets/exprs_map/test/pic/uncertainty_difference_{}.pdf".format(str(success)))

def detect_aberration(data, success, threshold = 0.2):
    sum_traj = 0
    exceed_traj = 0
    for i, traj in enumerate(data):
        sum_traj += 1
        if traj["SR"][0] == success:
            for k, v in traj["uncertainty"].items():
                if v > threshold:
                    exceed_traj += 1
                    break
    print(exceed_traj/sum_traj)
    # 成功的: 0.083
    # 失败的: 0.185

def draw_roc(data, diff=False):
    # y_labels = [0, 1, 1, 0, 1]
    # probs = [0.02, 0.003, 0.1, 0.0054, 0.19]
    # y_labels = []
    labels = []
    preds = []
    probs = []
    for i, traj in enumerate(data):
        for j, node in enumerate(traj["pred"]):
            viewpoint = node[0]
            label, pred = None, None
            if viewpoint in traj["gt"] and traj["gt"].index(viewpoint) + 1 < len(traj["gt"]):
                label = traj["gt"][traj["gt"].index(viewpoint)+1]
            if j + 1 < len(traj["pred"]):
                pred = traj["pred"][j+1][0]
            if label and pred:
                labels.append(label), preds.append(pred)
                if diff:
                    if j > 0:
                        probs.append(traj["uncertainty"][str(j)]-traj["uncertainty"][str(j-1)])
                    else:
                        probs.append(traj["uncertainty"][str(j)])
                else:
                    probs.append(traj["uncertainty"][str(j)])
    y_labels = [0 if labels[i] == preds[i] else 1 for i in range(len(labels))]

            
    fpr, tpr, thresholds = roc_curve(y_labels, probs)
    auc = roc_auc_score(y_labels, probs)
    print("Auc: {}".format(auc))
    # 0.644 
    # 如果用 uncertainty 的差值则为 0.559
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for Misclassification')
    plt.legend(loc="lower right")
    plt.show()
    # plt.savefig("./datasets/exprs_map/test/pic/ROC.pdf")

def draw_aupr(data, diff=False):
    # y_labels = [0, 1, 1, 0, 1]
    # probs = [0.02, 0.003, 0.1, 0.0054, 0.19]
    # y_labels = []
    labels = []
    preds = []
    probs = []
    for i, traj in enumerate(data):
        for j, node in enumerate(traj["pred"]):
            viewpoint = node[0]
            label, pred = None, None
            if viewpoint in traj["gt"] and traj["gt"].index(viewpoint) + 1 < len(traj["gt"]):
                label = traj["gt"][traj["gt"].index(viewpoint)+1]
            if j + 1 < len(traj["pred"]):
                pred = traj["pred"][j+1][0]
            if label and pred:
                labels.append(label), preds.append(pred)
                if diff:
                    if j > 0:
                        probs.append(traj["uncertainty"][str(j)]-traj["uncertainty"][str(j-1)])
                    else:
                        probs.append(traj["uncertainty"][str(j)])
                else:
                    probs.append(traj["uncertainty"][str(j)])
    y_labels = [0 if labels[i] == preds[i] else 1 for i in range(len(labels))]

            
    precision, recall, thresholds = precision_recall_curve(y_labels, probs)

    # baseline
    baseline_probs = np.random.rand(len(probs))
    precision_random, recall_random, _ = precision_recall_curve(y_labels, baseline_probs)
    aupr_random = auc(recall_random, precision_random)
    print("baseline aupr: {}".format(aupr_random))

    aupr = auc(recall, precision)
    print("aupr: {}".format(aupr))
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPR = {aupr:.2f})')
    plt.plot(recall_random, precision_random, color='red', lw=2, label=f'baseline PR curve (AUPR = {aupr_random:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    # plt.show()
    plt.savefig("./datasets/exprs_map/test/pic/AUPR.pdf")

if __name__ == "__main__":
    dir_name = "./datasets/exprs_map/test/preds"
    result = load_result(dir_name)
    # print(result)
    # draw_relation_unc_and_SR(result)
    # draw_relation_unc_and_spl(result)
    # show_unc_trending(result, True)
    # detect_aberration(result, True)
    # show_unc_difference(result, False)
    # draw_roc(result, diff=True)
    draw_aupr(result)
