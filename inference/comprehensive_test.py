'''
python comprehensive_test.py \
  --crack_config   <config.py> \
  --crack_checkpoint <checkpoint.pth> \
  --root_dir       <테스트 데이터 폴더> \
  --result_dir     <결과 저장 폴더>
'''

import os
import argparse
import mmcv
import numpy as np
import csv
import warnings
from tqdm import tqdm
from mmseg.apis import init_model, inference_model
import torch
from utils import inference_segmentor_sliding_window
from quantify_seg_results import quantify_crack_width_length
import cv2

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
warnings.filterwarnings("ignore", category=RuntimeWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='Sequential Inference and Analysis of Crack Detection with Per-Image Metrics')
    parser.add_argument('--crack_config', help='the config file to inference crack')
    parser.add_argument('--crack_checkpoint', help='the checkpoint file to inference crack')
    parser.add_argument('--root_dir', help='Root directory containing all datasets')
    parser.add_argument('--result_dir', help='Directory to save the comparison results')
    parser.add_argument('--gt_base_dir', default=None,
                        help='(optional) External GT root. GT 경로: '
                             '{gt_base_dir}/{gt_ext_subdir}/{split}/{stem}_gtFine_labelIds.png')
    parser.add_argument('--gt_ext_subdir', default='Dilation9_gtFine',
                        help='gt_base_dir 사용 시 GT 서브폴더명 (기본: Dilation9_gtFine). '
                             'CircDilation9_gtFine 등으로 지정 가능.')
    parser.add_argument('--gt_subdir', default='gtFine',
                        help='gt_base_dir 미지정 시 leftImg8bit를 대체할 GT 서브폴더명 '
                             '(기본: gtFine). ReDilation9_gtFine 비교 시 해당 폴더명으로 지정.')
    parser.add_argument('--alpha', type=float, default=0.8, help='the alpha value for blending')
    parser.add_argument('--rgb_to_bgr', action='store_true', help='convert rgb to bgr, if the model palette is written in rgb format')
    parser.add_argument('--overwrite_crack_palette', action='store_true', help='overwrite the crack palette with black and red')
    return parser.parse_args()

def find_test_directories(root_dir):
    test_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath.endswith(os.path.join('leftImg8bit', 'test')):
            test_dirs.append(dirpath)
    return test_dirs

def find_corresponding_file(base_filename, directory, suffix):
    target_filename = base_filename.replace('_leftImg8bit.png', suffix)
    full_path = os.path.join(directory, target_filename)
    return full_path if os.path.exists(full_path) else None

def calculate_metrics(gt_img, pred_img):
    TP = np.logical_and(gt_img == 1, pred_img == 1).sum()
    FP = np.logical_and(gt_img == 0, pred_img == 1).sum()
    FN = np.logical_and(gt_img == 1, pred_img == 0).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return TP, FP, FN, precision, recall, f1_score, iou

def create_corresponding_save_directory(img_path, root_dir, result_dir):
    rel_path = os.path.relpath(os.path.dirname(img_path), root_dir)
    save_dir = os.path.join(result_dir, rel_path)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def create_mask_save_directory(img_path, root_dir, result_dir):
    rel_path = os.path.relpath(os.path.dirname(img_path), root_dir)
    mask_rel_path = rel_path.replace('leftImg8bit', 'pred_gtFine')
    mask_save_dir = os.path.join(result_dir, mask_rel_path)
    os.makedirs(mask_save_dir, exist_ok=True)
    return mask_save_dir

def process_image(img_path, root_dir, result_dir, crack_model, model_args):
    base_filename = os.path.basename(img_path)
    if model_args.gt_base_dir:
        split  = os.path.basename(os.path.dirname(img_path))
        gt_dir = os.path.join(model_args.gt_base_dir, model_args.gt_ext_subdir, split)
    else:
        gt_dir = os.path.dirname(img_path).replace('leftImg8bit', model_args.gt_subdir)
    gt_path = find_corresponding_file(base_filename, gt_dir, '_gtFine_labelIds.png')

    if gt_path is None:
        return None

    # Perform inference
    _, crack_mask = inference_segmentor_sliding_window(crack_model, img_path, color_mask=None, score_thr=0.1, window_size=1024, overlap_ratio=0.1)

    # Save raw prediction mask (label values: 0, 1, 2, ...) before visualization
    save_dir = create_corresponding_save_directory(img_path, root_dir, result_dir)
    mask_save_dir = create_mask_save_directory(img_path, root_dir, result_dir)
    pred_label_filename = base_filename.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
    cv2.imwrite(os.path.join(mask_save_dir, pred_label_filename), crack_mask.astype(np.uint8))

    seg_result = mmcv.imread(img_path)
    gt_img = mmcv.imread(gt_path, flag='grayscale')

    # Visualize the crack mask
    crack_palette = crack_model.dataset_meta['palette'][:2]
    if model_args.rgb_to_bgr:
        crack_palette = [p[::-1] for p in crack_palette]
    if model_args.overwrite_crack_palette:
        crack_palette[1] = [0, 0, 255]

    color = np.array(crack_palette[1], dtype=np.uint8)
    mask_bool = crack_mask == 1

    seg_result[mask_bool] = seg_result[mask_bool] * (1 - model_args.alpha) + color * model_args.alpha

    # Quantify crack width and length
    seg_result = quantify_crack_width_length(seg_result, crack_mask, crack_palette[1])

    TP, FP, FN, precision, recall, f1_score, iou = calculate_metrics(gt_img, crack_mask)

    # Create comparison image
    comparison_img = np.zeros((*gt_img.shape, 3), dtype=np.uint8)

    # Visualize FP first (Magenta)
    comparison_img[np.logical_and(gt_img == 0, crack_mask == 1)] = [255, 0, 255]

    # Visualize FN second (Red)
    comparison_img[np.logical_and(gt_img == 1, crack_mask == 0)] = [0, 0, 255]

    # Visualize TP last, with increased size (Green)
    tp_mask = np.logical_and(gt_img == 1, crack_mask == 1)
    tp_mask_enlarged = cv2.dilate(tp_mask.astype(np.uint8), np.ones((3,3), np.uint8))
    comparison_img[tp_mask_enlarged == 1] = [0, 255, 0]

    combined_img = np.concatenate((seg_result, comparison_img), axis=1)

    # Save visualization with metrics in filename
    result_filename = f"p{precision:.4f}_r{recall:.4f}_f1{f1_score:.4f}_iou{iou:.4f}_{base_filename}"
    result_path = os.path.join(save_dir, result_filename)
    mmcv.imwrite(combined_img, result_path)

    return TP, FP, FN, precision, recall, f1_score, iou

def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    # Initialize model
    crack_model = init_model(args.crack_config, args.crack_checkpoint, device='cuda')

    test_dirs = find_test_directories(args.root_dir)

    total_TP, total_FP, total_FN = 0, 0, 0
    all_images = []

    for test_dir in test_dirs:
        for img_file in os.listdir(test_dir):
            if img_file.endswith('_leftImg8bit.png'):
                img_path = os.path.join(test_dir, img_file)
                all_images.append(img_path)

    print(f"Total images to process: {len(all_images)}")

    # Prepare CSV for per-image results
    per_image_csv_path = os.path.join(args.result_dir, 'per_image_results.csv')
    with open(per_image_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'Precision', 'Recall', 'F1-score', 'IoU'])

    for img_path in tqdm(all_images, desc="Processing images"):
        result = process_image(img_path, args.root_dir, args.result_dir, crack_model, args)
        if result:
            TP, FP, FN, precision, recall, f1_score, iou = result
            total_TP += TP
            total_FP += FP
            total_FN += FN

            # Write per-image results to CSV
            with open(per_image_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([os.path.basename(img_path), precision, recall, f1_score, iou])

    # Calculate global metrics
    global_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    global_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    global_f1_score = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
    global_iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0

    print(f"Total images processed: {len(all_images)}")
    print(f"Global Precision: {global_precision:.4f}")
    print(f"Global Recall: {global_recall:.4f}")
    print(f"Global F1-score: {global_f1_score:.4f}")
    print(f"Global IoU: {global_iou:.4f}")

    # Save global results to CSV
    global_csv_path = os.path.join(args.result_dir, 'global_results.csv')
    with open(global_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Global Precision', global_precision])
        writer.writerow(['Global Recall', global_recall])
        writer.writerow(['Global F1-score', global_f1_score])
        writer.writerow(['Global IoU', global_iou])

    print(f"Global results saved to: {global_csv_path}")
    print(f"Per-image results saved to: {per_image_csv_path}")

if __name__ == '__main__':
    main()
