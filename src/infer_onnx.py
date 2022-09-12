import os
import math
from glob import glob
import timeit
import yaml
from tqdm import tqdm

import onnxruntime as ort

from tempfile import NamedTemporaryFile
from datetime import datetime
from collections import deque, defaultdict
from warnings import warn

import cv2
import numpy as np

from hover_serving.src.scripts.viz_utils import visualize_instances, rm_n_mkdir 
import hover_serving.src.scripts.process_utils as proc_utils

### for local scripts
# from scripts.viz_utils import visualize_instances, rm_n_mkdir
# import scripts.process_utils as proc_utils


def timer(func, *args, **kwargs):
    start_time = timeit.default_timer()
    result = func(*args, **kwargs)
    elapsed = timeit.default_timer() - start_time
    print(f"Finished {func.__name__}. Time elapsed: {'{:.3f}'.format(elapsed)} sec")
    return result


class Inferer:
    def __init__(self, folder_img, model_path, providers=['CPUExecutionProvider'], batch_size=30, verbose=False):
        self.model_name = os.path.basename(model_path)

        if "consep" in self.model_name:
            self.mask_shape = [80, 80]
            self.input_shape = [270, 270]
            self.nuclei_types = {
                # Background: 0
                "Misc": 1,
                "Inflammatory": 2,
                "Epithelial": 3,
                "Spindle": 4,
            }
        elif "pannuke" in self.model_name:
            self.mask_shape = [164, 164]
            self.input_shape = [256, 256]
            self.nuclei_types = {
                # Background: 0
                "Inflammatory": 1,
                "Connective": 2,
                "Dead cells": 3,
                "Epithelial": 4,
                "Neoplastic cells": 5,
            }

        elif "monusac" in self.model_name:
            self.mask_shape = [164, 164]
            self.input_shape = [256, 256]
            self.nuclei_types = {
                "Epithelial": 1, 
                "Lymphocyte": 2, 
                "Macrophage": 3, 
                "Neutrophil": 4,
            }

        self.nr_types = len(self.nuclei_types.values()) + 1
        self.input_norm = True
        self.remap_labels = False
        self.inf_batch_size = batch_size

        # self.eval_inf_input_tensor_names = ["images:0"]
        # self.eval_inf_output_tensor_names = ["predmap-coded:0"]

        self.colors = {
            "Inflammatory": (0.0, 255.0, 0.0),  # bright green
            "Dead cells": (255.0, 255.0, 0.0),  # bright yellow
            "Neoplastic cells": (255.0, 0.0, 0.0),  # red  # aka Epithelial malignant
            "Epithelial": (0.0, 0.0, 255.0),  # dark blue  # aka Epithelial healthy
            "Misc": (0.0, 0.0, 0.0),  # pure black  # aka 'garbage class'
            "Spindle": (0.0, 255.0, 255.0),  # cyan  # Fibroblast, Muscle and Endothelial cells
            "Connective": (0.0, 220.0, 220.0),  # darker cyan    # Connective plus Soft tissue cells
            "Background": (255.0, 0.0, 170.0),  # pink
            ###
            "Lymphocyte": (170.0, 255.0, 0.0),  # light green
            "Macrophage": (170.0, 0.0, 255.0),  # purple
            "Neutrophil": (255.0, 170.0, 0.0),  # orange
            "black": (32.0, 32.0, 32.0),  # black
        }

        self.color_mapping = {v: k for k, v in self.nuclei_types.items()}
        for key in self.color_mapping:
            self.color_mapping[key] = self.colors[self.color_mapping[key]]

        self.img_paths = sorted(glob(f"{os.path.join(folder_img, '*.png')}"))
        self.predictor = ort.InferenceSession(model_path, providers=providers)
        if verbose: 
            print(f"Using {self.model_name} model")
            print(f"Selected provider(s): {self.predictor.get_providers()}")
            print(f"Using device: {ort.get_device()}")


    def __gen_prediction(self, x):

        step_size = self.mask_shape
        msk_size = self.mask_shape
        win_size = self.input_shape

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        im_h = x.shape[0]
        im_w = x.shape[1]

        last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
        last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h

        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w

        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), "reflect")

        sub_patches = []

        for row in range(0, last_h, step_size[0]):
            for col in range(0, last_w, step_size[1]):
                win = x[row : row + win_size[0], col : col + win_size[1]]
                sub_patches.append(win)

        pred_map = deque()

        while len(sub_patches) > self.inf_batch_size:
            mini_batch = sub_patches[: self.inf_batch_size]
            sub_patches = sub_patches[self.inf_batch_size :]
            mini_output = self.predictor.run(['predmap-coded:0'], {'images:0': mini_batch})[0]
            mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            mini_output = self.predictor.run(['predmap-coded:0'], {'images:0': sub_patches})[0]
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)

        # Assemble back into full image
        output_patch_shape = np.squeeze(pred_map[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        # Assemble back into full image
        pred_map = np.squeeze(np.array(pred_map))
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = (
            np.transpose(pred_map, [0, 2, 1, 3, 4])
            if ch != 1
            else np.transpose(pred_map, [0, 2, 1, 3])
        )
        pred_map = np.reshape(
            pred_map,
            (
                pred_map.shape[0] * pred_map.shape[1],
                pred_map.shape[2] * pred_map.shape[3],
                ch,
            ),
        )
        pred_map = np.squeeze(pred_map[:im_h, :im_w])  # just crop back to original size

        return pred_map

    def __process_instance(self, pred_map, output_dtype="uint16"):
        """
        Post processing script for image tiles

        Args:
            pred_map: commbined output of nc, np and hv branches
            output_dtype: data type of output

        Returns:
            pred_inst:     pixel-wise nuclear instance segmentation prediction
            pred_type_out: pixel-wise nuclear type prediction
        """

        # pred = np.squeeze(pred_map['result'])

        pred_inst = pred_map[..., self.nr_types :]
        pred_type = pred_map[..., : self.nr_types]

        pred_inst = np.squeeze(pred_inst)
        pred_type = np.argmax(pred_type, axis=-1)
        pred_type = np.squeeze(pred_type)

        pred_inst = proc_utils.proc_np_hv(pred_inst)

        if self.remap_labels:
            pred_inst = proc_utils.remap_label(pred_inst, by_size=True)

        pred_type_out = np.zeros([pred_type.shape[0], pred_type.shape[1]])
        # * Get class of each instance id, stored at index id-1
        pred_id_list = list(np.unique(pred_inst))[1:]  # exclude background ID
        pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
        for idx, inst_id in enumerate(pred_id_list):
            inst_tmp = pred_inst == inst_id
            inst_type = pred_type[pred_inst == inst_id]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            pred_type_out += inst_tmp * inst_type

            pred_inst_type[idx] = inst_type

        pred_type_out = pred_type_out.astype(output_dtype)
        pred_inst_out = pred_inst.astype(output_dtype)
        inst_type_out = pred_inst_type[:, None]
        # pred_inst_centroid = get_inst_centroid(pred_inst)

        pred = {'inst_map': pred_inst_out,
                'type_map': pred_type_out,
                'inst_type': inst_type_out,
        } # 'inst_centroid': pred_inst_centroid}
        return pred
    
    def run(self, process_dir='/tmp/', return_counts=True):
        time_tag = str(datetime.now().strftime("%m%d%y_%H%M%S_%f"))
        save_dir = os.path.join(process_dir, f'{time_tag}')

        rm_n_mkdir(save_dir)

        for img_path in tqdm(self.img_paths):
            filename = os.path.basename(img_path)
            basename = filename.split(".")[0]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred_map = self.__gen_prediction(img)
            pred = self.__process_instance(pred_map)
            overlaid_output = visualize_instances(img, self.color_mapping, pred['inst_map'], pred['type_map'])
            cv2.imwrite(os.path.join(save_dir, "{}.png".format(basename)), cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB))
            if return_counts:
                with open(os.path.join(save_dir, f'{basename}.log'), 'w') as log_file:
                    unique, counts = np.unique(pred['inst_type'], return_counts=True)
                    type_counts = dict(zip(unique, counts))
                    result = {}
                    for str_type, code in self.nuclei_types.items():
                        if code in type_counts.keys():
                            result[str_type] = type_counts[self.nuclei_types[str_type]]
                    print(result, file=log_file)