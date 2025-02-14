import warnings
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split

# Suppress all warnings
warnings.filterwarnings('ignore')


def preprocess_data(mode, train_path=None, test_path=None):
    if mode == 'train':
        # Load training dataset
        df_train = pd.read_csv(train_path)
        df = df_train.copy()
    elif mode == 'predict':
        # Load training and test datasets
        df_train = pd.read_csv(train_path)
        x_test_1 = pd.read_csv(test_path)
        session_ids_x_test_1 = x_test_1['session_id']
        df = pd.concat([df_train, x_test_1], ignore_index=True)
    else:
        raise ValueError("Mode must be either 'train' or 'predict'")

    print("Columns in dataframe before dropping:", df.columns.tolist())

    # Preprocessing
    df.drop(['product_category_2', 'city_development_index'], axis=1, inplace=True)
    if mode == 'train':
        df.loc[:, 'session_id'] = df['session_id'].where(
            ~df['session_id'].duplicated(keep='first') | df['session_id'].isna())
        df.dropna(subset=['is_click'], inplace=True)

    # Fill missing user_group_id values grouped by user_id
    labels = ['user_group_id']
    df[labels] = df.groupby('user_id')[labels].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

    # Fill missing user_group_id values by the underlying logic
    df['user_group_id'] = np.where(df['gender'] == 'Male', df['age_level'], df['age_level'] + 6)

    # Fill missing campaign_id values by the webpage_id values
    mapping_dict = df.dropna().set_index('webpage_id')['campaign_id'].to_dict()
    df['campaign_id'] = df['campaign_id'].fillna(df['webpage_id'].map(mapping_dict))


    if mode == 'predict':
        df['user_group_id'] = np.where(df['gender'] == 'Male', df['age_level'], df['age_level'] + 6)
        mapping_dict = x_test_1.dropna().set_index('webpage_id')['campaign_id'].to_dict()
        df['campaign_id'] = df['campaign_id'].fillna(df['webpage_id'].map(mapping_dict))

    if mode == 'train':
        df.dropna(thresh=len(df.columns) - 5, inplace=True)

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['day'] = df['DateTime'].dt.day
    df['hour'] = df['DateTime'].dt.hour
    df['total_sessions_per_user'] = df.groupby('user_id')['session_id'].transform('count')
    df['day_binary'] = (df['day'] > 4.5).astype(int)
    if mode == 'train':
        df.dropna(axis=0, subset=df.columns.difference(['is_click']), inplace=True)

    # Hour Bins
    bins = [(0, 6), (6, 12), (12, 18), (18, 24)]
    for start, end in bins:
        df[f'hour_{start}_{end}'] = ((df['hour'] >= start) & (df['hour'] < end)).astype(int)

    # Session bins
    session_bins = [(1, 5), (6, 10), (11, 20), (21, 50), (50, 100)]
    for start, end in session_bins:
        df[f'session_{start}_{end}'] = ((df['total_sessions_per_user'] >= start) &
                                        (df['total_sessions_per_user'] < end)).astype(int)
    df['session_100_plus'] = (df['total_sessions_per_user'] > 100).astype(int)

    # Define categorical columns for one-hot encoding
    categorical_cols = ['campaign_id', 'product_category_1', 'user_group_id', 'product'] + \
                       [f'session_{start}_{end}' for start, end in session_bins] + ['session_100_plus']

    # Scale day and hour features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[['day', 'hour']] = scaler.fit_transform(df[['day', 'hour']])


    # One-Hot Encoding (do not include is_click in predict mode)
    df_dummies = pd.get_dummies(
        df[categorical_cols + ['day', 'hour', 'day_binary']],
        columns=categorical_cols,
        drop_first=True
    )

    if mode == 'predict':
        # After one-hot encoding
        # Get the test portion of the dataframe
        x_test_dummies = df_dummies[df_dummies.index >= len(df_train)]

        # If 'is_click' exists (it might not in predict mode), drop it from the features.
        if 'is_click' in x_test_dummies.columns:
            x_test_dummies = x_test_dummies.drop(columns=['is_click'])

        # Prepare the feature columns by dropping 'is_click' if it exists
        if 'is_click' in df_dummies.columns:
            feature_columns = df_dummies.columns.drop('is_click')
        else:
            feature_columns = df_dummies.columns

        x_test_dummies = x_test_dummies.reindex(columns=feature_columns, fill_value=0)
        return x_test_dummies, session_ids_x_test_1

    if mode == "train":
        df_dummies['is_click'] = df['is_click']

    print("Columns in dataframe after one-hot encoding:", df_dummies.columns.tolist())

    if mode == 'train':
        return df_dummies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        df_train_processed = preprocess_data(args.mode, args.train_path)
    else:
        X_test_dummies, session_ids_x_test_1 = preprocess_data(args.mode, args.train_path, args.test_path)
    output_dir = os.path.dirname(args.train_path)

    if args.mode == "train":
        # Splitting data into train and test sets (80-20 split)
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(df_train_processed, test_size=0.2, random_state=42)
        # Saving train and test datasets
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        print(f"Train and test datasets saved to {output_dir}")
    elif args.mode == "predict":
        # Saving the prediction dataset
        X_test_dummies.to_csv(os.path.join(output_dir, "predict.csv"), index=False)
        print(f"Prediction dataset saved to {output_dir}")
