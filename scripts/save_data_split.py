import os
import pandas as pd
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
import datasets.data_utils as data_utils

def save_fold_splits(args,dataset_dict, save_dir, prefix='fold'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_folds = len(dataset_dict['train'])

    for fold_idx in range(n_folds):

        train_df = dataset_dict['train'][fold_idx]
        test_df = dataset_dict['test'][fold_idx]

        train_df = train_df.assign(Split='train') if 'Split' not in train_df.columns else train_df
        test_df = test_df.assign(Split='test') if 'Split' not in test_df.columns else test_df

        dfs_to_concat = [train_df, test_df]

        if dataset_dict.get('val') is not None and len(dataset_dict['val']) > 0:
            val_df = dataset_dict['val'][fold_idx]
            if val_df is not None and len(val_df) > 0:
                val_df = val_df.assign(Split='val') if 'Split' not in val_df.columns or args.append_val else val_df
                dfs_to_concat.append(val_df)

        fold_df = pd.concat(dfs_to_concat, axis=0).reset_index(drop=True)

        if n_folds == 1:
            save_path = save_dir / f'{prefix}.csv'  # 单次划分时不添加折号
        else:
            save_path = save_dir / f'{prefix}_{fold_idx}.csv'

        if not os.path.isfile(save_path):
            fold_df.to_csv(save_path, index=False)
            print(f"Saved {'fold ' + str(fold_idx + 1) if n_folds > 1 else 'split'} to {save_path}")

        print(f"{'Fold ' + str(fold_idx + 1) if n_folds > 1 else 'Split'} statistics:")
        print(f"Train samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")

        if dataset_dict.get('val') is not None and len(dataset_dict['val']) > 0:
            val_df = dataset_dict['val'][fold_idx]
            if val_df is not None and len(val_df) > 0:
                print(f"Val samples: {len(val_df)}")

        print(f"Total samples: {len(fold_df)}")
        print("-" * 50)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process CSV files and save cross-validation splits")
    parser.add_argument('--input_dir', type=str, default="xxx/dataset/wsi_data/label/labels_twh", help='Input directory containing CSV files, 可以是文件夹也可以是csv文件')
    parser.add_argument('--output_dir', type=str,  default="xxx/dataset/wsi_data/label/labels_twh_new", help='Output directory to save the splits, 必须是文件夹')
    parser.add_argument('--cv_fold', type=int, default=5, help='Number of cross-validation folds (default: 1)')
    parser.add_argument('--val_ratio', type=float, default=0.125, help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Validation set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed (default: 2021)')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--survival', action='store_true')
    parser.add_argument('--append_val', action='store_true')

    args = parser.parse_args()

    utils.seed_torch(args.seed)

    input_path = Path(args.input_dir)
    csv_files = []

    if input_path.is_file() and input_path.suffix == '.csv':
        csv_files = [input_path]
    elif input_path.is_dir():
        if args.cv_fold > 1 and args.append_val:
            fold_files = list(input_path.glob(f'fold_*.csv'))
            if len(fold_files) == 0:
                fold_files = list(input_path.glob(f'*.csv'))
            
            if len(fold_files) < args.cv_fold:
                print(f"Warning: Found {len(fold_files)} fold files, but expected {args.cv_fold}")
            
            csv_files = sorted(fold_files)[:args.cv_fold]
            
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            train_dfs, test_dfs, val_dfs = [], [], []
            
            for i, fold_file in enumerate(csv_files):
                print(f"Processing fold file {i+1}/{len(csv_files)}: {fold_file}")
                
                if 'survival' in str(fold_file) or args.survival:
                    fold_df = data_utils.get_patient_label_surv(None, fold_file)
                else:
                    fold_df = data_utils.get_patient_label(None, fold_file)

                split_col = 'Split' if 'Split' in fold_df.columns else 'split'
                if split_col not in fold_df.columns:
                    raise ValueError(f"Fold file {fold_file} does not contain a 'Split' or 'split' column")

                train_df = fold_df[fold_df[split_col].str.lower() == 'train'].copy()
                test_df = fold_df[fold_df[split_col].str.lower() == 'test'].copy()
                
                if args.val_ratio > 0:
                    val_df, train_df = data_utils.data_split(args.seed + i, train_df, args.val_ratio)
                else:
                    val_df = None
                
                train_dfs.append(train_df)
                test_dfs.append(test_df)
                val_dfs.append(val_df)
            
            dataset = {
                'train': train_dfs,
                'test': test_dfs,
                'val': val_dfs
            }
            
            save_fold_splits(args, dataset, output_dir, prefix='fold')
            
            sys.exit(0)
        else:
            csv_files = list(input_path.glob('*.csv'))
    else:
        raise ValueError("Input path must be a CSV file or a directory containing CSV files")

    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")

        if args.cv_fold > 1:
            output_dir = os.path.join(args.output_dir, csv_file.stem)
        else:
            output_dir = args.output_dir
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        if 'survival' in str(csv_file) or args.survival:
            df = data_utils.get_patient_label_surv(None, csv_file)
        else:
            df = data_utils.get_patient_label(None, csv_file)

        if args.cv_fold > 1:
            if args.append_val:
                pass 
            else:

                if 'Split' in df.columns:
                    df = df.drop(columns=['Split'])
                if 'split' in df.columns:
                    df = df.drop(columns=['split'])

                df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
                # ONLY for compatible API
                args.val2test=False

                train_dfs, test_dfs, val_dfs = data_utils.get_kfold(
                args, 
                args.cv_fold, 
                df, 
                val_ratio=args.val_ratio
                )
        else:
            if args.append_val and ('Split' in df.columns or 'split' in df.columns):

                split_col = 'Split' if 'Split' in df.columns else 'split'
                
                train_df = df[df[split_col].str.lower() == 'train'].copy()
                test_df = df[df[split_col].str.lower() == 'test'].copy()

                if args.val_ratio > 0:
                    val_df, train_df = data_utils.data_split(args.seed, train_df, args.val_ratio)
                else:
                    val_df = None
            else:
                if 'Split' in df.columns:
                    df = df.drop(columns=['Split'])
                if 'split' in df.columns:
                    df = df.drop(columns=['split'])

                df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
                test_df, train_df = data_utils.data_split(args.seed, df, args.test_ratio)
                if args.val_ratio > 0:
                    val_df, train_df = data_utils.data_split(args.seed, train_df, args.val_ratio)
                else:
                    val_df = None

            train_dfs = [train_df]
            test_dfs = [test_df]
            val_dfs = [val_df]
        dataset = {
            'train': train_dfs,
            'test': test_dfs,
            'val': val_dfs
        }

        if args.cv_fold > 1:
            save_fold_splits(args,dataset, output_dir, prefix='fold')
        else:
            _title = args.title or csv_file.stem
            save_fold_splits(args,dataset,output_dir,prefix=_title)