
import os
import time
import datetime
import argparse
import logging
import configs
import json

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
from zeronlg import CaptionDataset
from zeronlg.utils import get_formatted_string, coco_caption_eval
from tqdm import tqdm

import argparse
import torch
import clip
from model.ZeroCLIP import CLIPTextGenerator


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

try:
    ROOT = configs.annotation_caption_root
except:
    ROOT = configs.annotation_root


def run(model, imgs, num_beams, cond_text="Image of a"):
    image_features = model.get_img_feature(None, None, imgs)
    captions = model.run(image_features, cond_text, beam_size=num_beams)

    encoded_captions = [model.clip.encode_text(clip.tokenize(c).to(model.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()
    return captions[best_clip_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data paths and attributes
    parser.add_argument('--data_root', type=str, default=ROOT)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--val_file', type=str, help='If not specified, use val_file_format')
    parser.add_argument('--val_gt_file', type=str, help='If not specified, use val_gt_file_format')
    parser.add_argument('--test_file', type=str, help='If not specified, use test_file_format')
    parser.add_argument('--test_gt_file', type=str, help='If not specified, use test_gt_file_format')
    parser.add_argument('--val_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/val.json'))
    parser.add_argument('--val_gt_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/val_gt.json'))
    parser.add_argument('--test_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/test.json'))
    parser.add_argument('--test_gt_file_format', type=str, default=os.path.join(ROOT, '{dataset}/{lang}/test_gt.json'))
    
    # Evaluation settings
    parser.add_argument('--modes', type=str, nargs='+', default=['test'], help='evaluation modes: ["val"], ["test"], ["val", "test"]')
    parser.add_argument('--lang', type=str, default='en', help='which language to be generated?')
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--num_frames', type=int, default=configs.num_frames)

    # Output settings
    parser.add_argument('--output_path', type=str, default='output/zerocap')

    # ZeroCLIP
    parser.add_argument("--clip_checkpoints", type=str, default="./data/checkpoints/", help="path to CLIP")
    parser.add_argument("--gpt2_model", type=str, default="./data/checkpoints/gpt2-medium/")
    args = parser.parse_args()

    output_path = os.path.join(args.output_path, 'evaluations_caption', args.dataset, args.lang)
    os.makedirs(output_path, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(output_path, 'log.txt'), 'w', encoding='utf-8'))
    logger.info(f'output path: {output_path}')

    assert args.modes in [['val'], ['test'], ['val', 'test']]

    logger.info(f'Creating ZeroCLIP from {args.clip_checkpoints} and {args.gpt2_model}')
    model = CLIPTextGenerator(
        clip_checkpoints=args.clip_checkpoints,
        gpt2_model=args.gpt2_model,
    )

    # start evaluation
    start_time = time.time()
    for mode in args.modes:
        ann_rpath = get_formatted_string(vars(args), f"{mode}_file", assigned_keys=['dataset', 'lang'])
        logger.info(f'Load dataset from {ann_rpath}')

        dataset = CaptionDataset(
            vision_root=configs.image_video_root[args.dataset],
            ann_rpath=ann_rpath,
            num_frames=args.num_frames,
            lang=args.lang,
            logger=logger,
            return_images=True,
        )
        logger.info(f'There are {len(dataset)} vision inputs')

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        gt_file_path = get_formatted_string(vars(args), f"{mode}_gt_file", assigned_keys=['dataset', 'lang'])
        result_file = os.path.join(output_path, f'{mode}_captions.json')
        detailed_scores_file = os.path.join(output_path, f'{mode}_detailed_scores.json')
        scores_file = os.path.join(output_path, f'{mode}_scores.json')

        results = []
        for batch in tqdm(loader):
            image_ids, images = batch
            
            image_id = image_ids[0]
            imgs = images[0]
            assert len(imgs) in [1, args.num_frames]

            caption = run(model, imgs, args.num_beams)
            results.append({"image_id": image_id, "caption": caption})
            print(image_id, caption)
        
        logger.info(f'Save caption results to {result_file}')
        json.dump(results, open(result_file, 'w'))

        coco_test = coco_caption_eval(gt_file_path, result_file, eval_lang=args.lang)
            
        logger.info(f'Save detailed scores to {detailed_scores_file}')
        json.dump(coco_test.evalImgs, open(detailed_scores_file, 'w'))

        logger.info(f'Save scores to {scores_file}')
        json.dump(coco_test.eval, open(scores_file, 'w'))

        for k, v in coco_test.eval.items():
            logger.info(f'[{mode}] {k} {v}')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Time {}'.format(total_time_str))

