#!/usr/bin/env python
"""
For evaluation
Extended from ADNet code by Hansen et al.
"""
import shutil
import SimpleITK as sitk
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from models.FoB import FewShotSeg
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import *
from utils import *
from config import ex
from SAM import SAM
from dataloaders.isic_setting_1 import (
    get_isic_setting_1_categories,
    get_superpix_isic_dataset,
)
from dataloaders.isic_setting_2 import DatasetISIC
import torchvision.transforms as transforms
@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproduciablity.
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model_config = {
    }
    model = FewShotSeg(model_config)
    model.cuda()
    model.load_state_dict(torch.load(_config['reload_model_path'], map_location='cpu'), strict=False)
    sam = SAM(sam_pretrained_path=".../sam_vit_h_4b8939.pth")
    _log.info(f'Load data...')
    if _config['dataset'] == 'isic':
        if _config['isic_setting'] == 1:
            base_path = _config['isic_setting_1_base_path']
            categories = get_isic_setting_1_categories(base_path)

            global_dice_sum = 0.0
            global_iou_sum = 0.0
            global_sample_count = 0

            model.eval()

            for cat in categories:
                _log.info(f'===== Evaluating Category {cat} =====')

                test_dataset = get_superpix_isic_dataset(
                    base_path=base_path,
                    fold=_config['eval_fold'],
                    category=cat,
                    train=False
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=_config['batch_size'],
                    shuffle=False,
                    num_workers=_config['num_workers'],
                    pin_memory=True,
                    drop_last=False
                )

                scores = Scores()
                dice_sum = 0.0
                iou_sum = 0.0
                sample_count = 0

                with torch.no_grad():
                    for _, sample in enumerate(test_loader):
                        support_images = [[shot.float().cuda() for shot in way]
                                          for way in sample['support_images']]
                        support_fg_mask = [[shot.float().cuda() for shot in way]
                                           for way in sample['support_fg_labels']]

                        query_images = [query_image.float().cuda() for query_image in sample['query_images']]
                        query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)

                        neg_point, pos_point = model(
                            support_images,
                            support_fg_mask,
                            query_images,
                            query_labels,
                            False
                        )

                        pred = sam(
                            query_images[0][0],
                            pos_point,
                            neg_point,
                            _config,
                            return_logits=False
                        )

                        pred = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
                        scores.record(pred, query_labels.cpu())

                        dice_sum += scores.patient_dice[-1]
                        iou_sum += scores.patient_iou[-1]
                        sample_count += 1

                if sample_count > 0:
                    dice_mean = dice_sum / sample_count
                    iou_mean = iou_sum / sample_count
                else:
                    dice_mean = torch.tensor(0.0)
                    iou_mean = torch.tensor(0.0)

                _log.info(f'Category {cat} Dice: {dice_mean.item()}')
                _log.info(f'Category {cat} IoU: {iou_mean.item()}')

                global_dice_sum += dice_sum
                global_iou_sum += iou_sum
                global_sample_count += sample_count

            global_dice = global_dice_sum / global_sample_count
            global_iou = global_iou_sum / global_sample_count

            _log.info('===== Mean Results =====')
            _log.info(f'Mean Dice: {global_dice.item()}')
            _log.info(f'Mean IoU: {global_iou.item()}')

            with open('results.txt', 'w') as file:
                file.write(f'Dice: {global_dice.item()}\n')
                file.write(f'IoU: {global_iou.item()}\n')

            return 1

        elif _config['isic_setting'] == 2:
            img_mean = [0.485, 0.456, 0.406]
            img_std = [0.229, 0.224, 0.225]

            transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(img_mean, img_std)])
            test_dataset = DatasetISIC(
                datapath=_config['isic_setting_2_base_path'],
                fold=_config['eval_fold'],
                transform=transform,
                split='test',
                shot=1
            )
            test_loader = DataLoader(test_dataset,
                                 batch_size=_config['batch_size'],
                                 shuffle=False,
                                 num_workers=_config['num_workers'],
                                 pin_memory=True,
                                 drop_last=True)
            scores = Scores()
            model.eval()
            Dice = 0.0
            IoU = 0.0
            counter = 0
            with torch.no_grad():
                for _, sample in enumerate(test_loader):

                    # Prepare episode data.
                    support_images = [[shot.float().cuda() for shot in way]
                                    for way in sample['support_images']]
                    support_fg_mask = [[shot.float().cuda() for shot in way]
                                    for way in sample['support_fg_labels']]

                    query_images = [query_image.float().cuda() for query_image in sample['query_images']]
                    query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)


                    # Forward pass
                    neg_point, pos_point = model(support_images, support_fg_mask, query_images, query_labels, False)

                    pred = sam(query_images[0][0], pos_point, neg_point, _config, return_logits=False)


                    pred = torch.from_numpy(pred).float().cuda().unsqueeze(0).unsqueeze(0).cpu()
                    # Calculate loss
                    scores.record(pred, query_labels.cpu())

                    Dice += scores.patient_dice[-1]
                    IoU += scores.patient_iou[-1]
                    counter += 1

            Dice /= counter
            IoU /= counter
            _log.info(f'Mean Dice: {Dice.item()}')
            _log.info(f'Mean IoU: {IoU.item()}')


            with open('results.txt', 'w') as file:
                file.write(str(Dice.item()))

            return 1

        else:
            raise ValueError(f'Unsupported ISIC setting: {_config["isic_setting"]}')

    
    else:
        data_config = {
            'data_dir': _config['path'][_config['dataset']]['data_dir'],
            'dataset': _config['dataset'],
            'n_shot': _config['n_shot'],
            'n_way': _config['n_way'],
            'n_query': _config['n_query'],
            'n_sv': _config['n_sv'],
            'max_iter': _config['max_iters_per_load'],
            'eval_fold': _config['eval_fold'],
            'min_size': _config['min_size'],
            'max_slices': _config['max_slices'],
            'supp_idx': _config['supp_idx'],
        }
        test_dataset = TestDataset(data_config)
    test_loader = DataLoader(test_dataset,
                             batch_size=_config['batch_size'],
                             shuffle=True,
                             num_workers=_config['num_workers'],
                             pin_memory=True,
                             drop_last=True)

    # Get unique labels (classes).
    labels = get_label_names(_config['dataset'])

    # Loop over classes.
    class_dice = {}
    class_iou = {}


    _log.info(f'Starting validation...')
    for label_val, label_name in labels.items():

        # Skip BG class.
        if label_name == 'BG':
            continue
        elif np.intersect1d([label_val], _config['test_label']).size == 0:
            continue

        _log.info(f'Test Class: {label_name}')

        # Get support sample + mask for current class.
        support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=_config['n_part'])


        test_dataset.label = label_val

        # Test.
        with torch.no_grad():
            model.eval()

            # Unpack support data.
            support_image = [support_sample['image'][[i]].float().cuda() for i in
                             range(support_sample['image'].shape[0])]  # n_shot x 3 x H x W, support_image is a list {3X(1, 3, 256, 256)}
            support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in
                               range(support_sample['image'].shape[0])]  # n_shot x H x W

            # Loop through query volumes.
            scores = Scores()
            for i, sample in enumerate(test_loader):  # this "for" loops 4 times

                # Unpack query data.
                query_image = [sample['image'][i].float().cuda() for i in
                               range(sample['image'].shape[0])]  # [C x 3 x H x W] query_image is list {(C x 3 x H x W)}
                query_label = sample['label'].long()  # C x H x W
                query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]


                # Compute output.
                # Match support slice and query sub-chunck.
                query_pred = torch.zeros(query_label.shape[-3:])
                C_q = sample['image'].shape[1]    # slice number of query img

                idx_ = np.linspace(0, C_q, _config['n_part'] + 1).astype('int')
                for sub_chunck in range(_config['n_part']):  # n_part = 3
                    support_image_s = [support_image[sub_chunck]]  # 1 x 3 x H x W
                    support_fg_mask_s = [support_fg_mask[sub_chunck]]  # 1 x H x W
                    query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]  # C' x 3 x H x W
                    query_label_s = query_label[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]  # C' x H x W
                    # print(support_fg_mask_s[0].shape)
                    query_pred_s = []
                    for i in range(query_image_s.shape[0]):
                        neg_point, pos_point = model([support_image_s], [support_fg_mask_s], [query_image_s[[i]]],
                                           query_label_s[[i]],None)  # 1 x 2 x H x W

                        pred = sam([query_image_s[[i]]][0][0], pos_point, neg_point, _config, return_logits=False)
                        pred = torch.from_numpy(pred).float().cuda().unsqueeze(0).unsqueeze(0)
                        query_pred_s.append(pred)


                    query_pred_s = torch.cat(query_pred_s, dim=0)

                    query_pred_s = query_pred_s.squeeze(1)
                    query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_s



                query_pred = query_pred.cpu()
                query_label = query_label.cpu()
                # Record scores.
                scores.record(query_pred, query_label)


                # Log.
                _log.info(
                    f'Tested query volume: {sample["id"][0][len(_config["path"][_config["dataset"]]["data_dir"]):]}.')
                _log.info(f'Dice score: {scores.patient_dice[-1].item()}')


                # Save predictions.
                file_name = os.path.join(f'{_run.observers[0].dir}/interm_preds',
                                         f'prediction_{query_id}_{label_name}.nii.gz')
                itk_pred = sitk.GetImageFromArray(query_pred)
                sitk.WriteImage(itk_pred, file_name, True)
                _log.info(f'{query_id} has been saved. ')

            # Log class-wise results
            class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
            class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
            _log.info(f'Test Class: {label_name}')
            _log.info(f'Mean class IoU: {class_iou[label_name]}')
            _log.info(f'Mean class Dice: {class_dice[label_name]}')

    _log.info(f'Final results...')
    _log.info(f'Mean IoU: {class_iou}')
    _log.info(f'Mean Dice: {class_dice}')



    def dict_Avg(Dict):
        L = len(Dict)  
        S = sum(Dict.values())  
        A = S / L
        return A

    value = dict_Avg(class_dice)

    with open('results.txt', 'w') as file:
        file.write(str(value))


    _log.info(f'Whole mean Dice: {dict_Avg(class_dice)}')

    _log.info(f'End of validation.')
    return 1
