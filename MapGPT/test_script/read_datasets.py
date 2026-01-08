import json
import os

def read_traj(file_name):
    with open(file_name, "r") as file:
        json_obj = json.load(file)
        print(json_obj[0])
        len_dataset = len(json_obj)
        print(len_dataset)

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

if __name__ == "__main__":
    file = "./datasets/R2R/annotations/MapGPT_72_scenes_processed.json"
    read_traj(file)
