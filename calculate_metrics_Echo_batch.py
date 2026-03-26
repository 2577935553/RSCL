import os

import nibabel as nib
import numpy as np
import pandas as pd

from utilities import image_utils
# from image_utils import np_categorical_dice_3d, np_categorical_jaccard_3d, np_categorical_assd_hd

data_directory = os.getcwd()
print(data_directory)

pct_list=[5,10,20]
for pct in pct_list:
    path_data='/public/home/zhsy/data/zhsy_data'
    pre_name=f'RSCL/CAMUS_{pct}pct'
    pred_data_path=os.path.join(path_data, pre_name, 'predictions')
    num_classes = 4



    name_list = pd.read_csv(os.path.join(data_directory, 'data_dir/test_Echo_4CH.csv'))['label_filenames'].tolist()
    gt_label_list = ['database_nifti/{}'.format(i,i) for i in name_list]
    # pred_list = ['{}'.format(i,i).replace('gt','Pred') for i in name_list]
    pred_list = ['{}'.format(i,i).replace('gt_resamp','resamp_Pred') for i in name_list]
    output_file = f'Pred_Res_{pre_name.split("/")[-1]}.csv'
    # pred_list = ['{}/PredAvgCMF.nii.gz'.format(i,i) for i in name_list]
    # output_file = 'PredAvgCMF_Res.csv'
    # pred_list = ['{}/PredAvgCMF_orig_space.nii.gz'.format(i,i) for i in name_list]
    # output_file = 'PredAvgCMF_orig_Res.csv'
    # pred_list = ['{}/MLKC_out.nii.gz'.format(i,i) for i in name_list]
    # output_file = 'MLKC_Res.csv'
    # pred_list = ['{}/MLKC_4DAvg.nii.gz'.format(i,i) for i in name_list]
    # output_file = 'MLKC_4DAvg_Res.csv'
    # pred_list = ['{}/MLKC_4Dout.nii.gz'.format(i,i) for i in name_list]
    # output_file = 'MLKC_4D_Res.csv'
    # pred_list = ['{}/MLKC_4D_4DAvg.nii.gz'.format(i,i) for i in name_list]
    # output_file = 'MLKC_4D_4DAvg_Res.csv'
    # pred_list = ['{}/MLKC_4D_4Dout.nii.gz'.format(i,i) for i in name_list]
    # output_file = 'MLKC_4D_4D_Res.csv'
    # pred_list = ['{}/MLKC_4D_4D_2Dout.nii.gz'.format(i,i) for i in name_list]
    # output_file = 'MLKC_4D_4D_2D_Res.csv'
    # pred_list = ['{}/MLKC_4D_4D_2DoutAvg.nii.gz'.format(i,i) for i in name_list]
    # output_file = 'MLKC_4D_4D_2DAvg_Res.csv'


    dice_classes = ['DiceClass{}'.format(i) for i in range(1, num_classes)]
    assd_np_classes = ['AssdNpClass{}'.format(i) for i in range(1, num_classes)]
    hd_np_classes = ['HdNpClass{}'.format(i) for i in range(1, num_classes)]
    # jaccard_classes = ['JaccardClass{}'.format(i) for i in range(1, num_classes)]
    volE_classes = ['VolEClass{}'.format(i) for i in range(1, num_classes)]
    volP_classes = ['VolPClass{}'.format(i) for i in range(1, num_classes)]
    volPred_classes = ['VolPredClass{}'.format(i) for i in range(1, num_classes)]
    volGT_classes = ['VolGTClass{}'.format(i) for i in range(1, num_classes)]

    results_list = []

    for gt_label_filename, pred_filename in zip(gt_label_list, pred_list):
        try:
            print('Processing GT: {}, Prediction: {}'.format(gt_label_filename, pred_filename))

            label_nii = nib.load(os.path.join(path_data, gt_label_filename))
            pixel_spacing = label_nii.header['pixdim'][1:4]
            label = label_nii.get_fdata().astype(np.int8).squeeze()

            pred_nii = nib.load(os.path.join(pred_data_path, pred_filename))
            pred = pred_nii.get_fdata().astype(np.int8).squeeze()

            dice = image_utils.np_categorical_dice_3d(pred, label, num_classes)
            if dice[1]>0.1:
                assd_np, hd_np = image_utils.np_categorical_assd_hd(pred, label, num_classes, pixel_spacing[0:2])

                subject_slice_df = {}
                subject_slice_df['Subject'] = gt_label_filename

                for j in range(1, num_classes):
                    subject_slice_df['DiceClass{}'.format(j)] = dice[j]
                    subject_slice_df['AssdNpClass{}'.format(j)] = assd_np[j]
                    subject_slice_df['HdNpClass{}'.format(j)] = hd_np[j]
                
                results_list.append(subject_slice_df)
        except Exception as e:
            print(f'Error processing {gt_label_filename}: {e}')
            continue

    df = pd.DataFrame(results_list)

    # 4. 计算统计摘要
    average = df.mean(axis=0, skipna=True, numeric_only=True)
    average['Subject'] = 'Mean'

    stddev = df.std(axis=0, ddof=0, skipna=True, numeric_only=True)
    stddev['Subject'] = 'SD'

    max_vals = df.max(axis=0, skipna=True)
    max_vals['Subject'] = 'Max'

    min_vals = df.min(axis=0, skipna=True)
    min_vals['Subject'] = 'Min'

    summary_df = pd.DataFrame([average, stddev, max_vals, min_vals])
    df = pd.concat([df, summary_df], ignore_index=True)

    df = df.round(4)
    df.to_csv(output_file, index=False, na_rep='nan')
    df.to

    print(f"\n处理完成，结果已保存到 {output_file}")
