import sys
sys.path.append('../../MambaCD')

import argparse
import os
import torch
import logging
import numpy as np
import json
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from skimage.measure import label, regionprops
from tqdm import tqdm
from changedetection.configs.config import get_config
from changedetection.datasets.make_data_loader import DamageAssessmentDatset
from changedetection.models.STMambaBDA import STMambaBDA
import imageio

logging.basicConfig(level=logging.INFO)

ori_label_value_dict = {
    'background': (0, 0, 0),         # Black
    'no_damage': (255, 0, 0),       # Red
    'minor_damage': (0, 255, 0),    # Green
    'major_damage': (0, 0, 255),    # Blue
    'destroy': (255, 255, 0)        # Yellow
}

target_label_value_dict = {
    'background': 0,
    'no_damage': 1,
    'minor_damage': 2,
    'major_damage': 3,
    'destroy': 4,
}

# Map labels to colors
def map_labels_to_colors(labels, ori_label_value_dict, target_label_value_dict):
    target_to_ori = {v: k for k, v in target_label_value_dict.items()}
    H, W = labels.shape
    color_mapped_labels = np.zeros((H, W, 3), dtype=np.uint8)

    for target_label, ori_label in target_to_ori.items():
        mask = labels == target_label
        color_mapped_labels[mask] = ori_label_value_dict[ori_label]
    
    return color_mapped_labels

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)
        self.deep_model = STMambaBDA(
            output_building=2, output_damage=5,
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        self.deep_model = self.deep_model.cuda()

        self.building_map_T1_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'building_localization_map')
        self.change_map_T2_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'damage_classification_map')
        self.confidence_map_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'confidence_map')

        if not os.path.exists(self.building_map_T1_saved_path):
            os.makedirs(self.building_map_T1_saved_path)
        if not os.path.exists(self.change_map_T2_saved_path):
            os.makedirs(self.change_map_T2_saved_path)
        if not os.path.exists(self.confidence_map_saved_path):
            os.makedirs(self.confidence_map_saved_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.deep_model.eval()

    def infer(self):
        torch.cuda.empty_cache()
        dataset = DamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)

        predictions_dict = {}

        confidence_threshold = self.args.conf_threshold  # Confidence threshold for filtering

        with torch.no_grad():
            for itera, data in enumerate(tqdm(val_data_loader)):
                pre_change_imgs, post_change_imgs, names = data
                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()

                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

                # Convert logits to probabilities
                probabilities_loc = softmax(output_loc.cpu(), dim=1).numpy()
                confidence_scores = np.max(probabilities_loc, axis=1)  # Max confidence per pixel

                # Get class predictions
                output_loc = np.argmax(output_loc.cpu().numpy(), axis=1)
                output_clf = np.argmax(output_clf.cpu().numpy(), axis=1)

                image_name = names[0] + '.png' if '.png' not in names[0] else names[0]

                # Log initial number of masks
                labeled_output = label(output_loc > 0)
                regions = regionprops(labeled_output)
                logging.info(f"Image: {image_name} - Masks before filtering: {len(regions)}")
                print(len(regions))

                num_masks_after = 0
                damage_count = {k: 0 for k in target_label_value_dict.keys()}

                for region in regions:
                    building_mask = (labeled_output == region.label)
                    region_confidence = confidence_scores[building_mask].mean()

                    if region_confidence < confidence_threshold:
                        continue  # Skip low-confidence regions

                    num_masks_after += 1
                    building_damage_classes = output_clf[building_mask]
                    most_common_class = np.bincount(building_damage_classes).argmax()

                    index_to_label = {v: k for k, v in target_label_value_dict.items()}
                    damage_label = index_to_label[most_common_class]
                    damage_count[damage_label] += 1

                logging.info(f"Image: {image_name} - Masks after filtering: {num_masks_after}")
                print(f"num_masks_output: {num_masks_after}")

                predictions_dict[image_name] = damage_count

                # Save visualizations and confidence map
                output_loc_image = (np.squeeze(output_loc) > 0).astype(np.uint8) * 255
                output_clf_image = map_labels_to_colors(np.squeeze(output_clf), ori_label_value_dict, target_label_value_dict)
                output_clf_image[output_loc == 0] = 0

                confidence_map_image = (confidence_scores * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(self.building_map_T1_saved_path, image_name), output_loc_image)
                imageio.imwrite(os.path.join(self.change_map_T2_saved_path, image_name), output_clf_image)
                imageio.imwrite(os.path.join(self.confidence_map_saved_path, image_name), confidence_map_image)

        # Save predictions
        with open('predictions.json', 'w') as json_file:
            json.dump(predictions_dict, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Inference on xBD dataset")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/MambaCD/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='xBD')
    parser.add_argument('--test_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
    parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_type', type=str, default='MambaBDA_Tiny')
    parser.add_argument('--result_saved_path', type=str, default='../results')
    parser.add_argument("--conf_threshold", type=float, default=0.0, help="conf Threshold to exclude small regions by mean confidence")
    # Add other arguments here as needed
    parser.add_argument('--resume', type=str)

    args = parser.parse_args()

    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.infer()

if __name__ == "__main__":
    main()
