import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(npz_path, node_id=1, history_size=12, future_size=1, step_size=1,
                          test_size=0.2, batch_size=128):
    # 加载数据
    raw = np.load(npz_path)['data']

    # 取出 node_id 节点的所有3个特征
    df = pd.DataFrame(raw[:, node_id, :], columns=['flow', 'occupy', 'speed'])

    target = df['flow'].values   # 只预测流量flow特征
    print("Total time steps: {}".format(len(target)))
    print(df.head())

    # 保存原始数据到 outputs
    df.to_csv(r'outputs/traffic_data.csv', index=False)
    print(" save to outputs/traffic_data.csv")


    features, labels = [], []
    for i in range(history_size, len(target) - future_size, step_size):
        features.append(target[i - history_size:i])
        labels.append(target[i:i + future_size])

    features = np.array(features).reshape(-1, history_size, 1)  # (样本数, history_size=3, input_size=1)
    labels = np.array(labels).reshape(-1, 1)
    print("Samples: {}, Shape: {}".format(len(features), features.shape))


    split_idx = int(len(features) * (1 - test_size))  #前80%的数据做xtrain
    x_train, x_test = features[:split_idx], features[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    #防止数据泄露
    x_train = scaler_x.fit_transform(x_train.reshape(-1, 1)).reshape(-1, history_size, 1) #标准化
    x_test = scaler_x.transform(x_test.reshape(-1, 1)).reshape(-1, history_size, 1)

    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # DataLoader
    train_dataset = Data.TensorDataset(x_train_t, y_train_t)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    return train_loader, x_test_t, y_test_t, scaler_y


if __name__ == '__main__':
    load_data = load_and_prepare_data(r'../dataset/pems04.npz')
