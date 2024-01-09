import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

from load_data import *
from model import EEGNet
from calculate import calculate_metrics

# 在循环开始前初始化 DataFrame
columns = ['Subject', 'Accuracy', 'AUC', 'Balanced Accuracy', 'Recall', 'Precision', 'F1 Score']
results_df = pd.DataFrame(columns=columns)
# 创建 results 文件夹，如果不存在的话
if not os.path.exists('results'):
    os.makedirs('results')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
# 设置被试列表
subject_list = [f"sub{i}{group}" for i in range(1, 4) for group in ["A", "B"]]

# 循环处理每个被试的数据
for subject in subject_list:
    # 构建数据文件和标签文件路径
    data_file_path = os.path.join("E:/Users/mengxu/桌面/EEGNet_EEG_RSVP/EEGNet/Data/qinghuamat/data", f"data_{subject}.mat")
    label_file_path = os.path.join("E:/Users/mengxu/桌面/EEGNet_EEG_RSVP/EEGNet/Data/qinghuamat/label", f"label_{subject}.mat")
    print(f"Loading subject {subject}...")
    # 在这里添加代码来加载MAT文件并处理数据
    data, label = load_data_and_labels(data_file_path, label_file_path)
    # 转换标签: 2 -> 0, 1 -> 1
    label = np.where(label == 2, 0, label)  # 将标签中的 2 替换为 0
    label = np.where(label == 1, 1, label)  # 将标签中的 1 保持为 1
    print(f"{subject} loaded and processed.")

    # 使用前4000个样本作为训练集，后4000个样本作为测试集
    train_data, test_data = data[:4000], data[4000:]
    train_labels, test_labels = label[:4000], label[4000:]
    # 将数据转换为 PyTorch Tensor
    tensor_X_train = torch.Tensor(train_data)
    tensor_Y_train = torch.Tensor(train_labels).long().squeeze()  # 使用 .squeeze() 确保标签是一维的
    tensor_X_test = torch.Tensor(test_data)
    tensor_Y_test = torch.Tensor(test_labels).long().squeeze()

    # 创建 TensorDataset
    train_dataset = TensorDataset(tensor_X_train, tensor_Y_train)
    test_dataset = TensorDataset(tensor_X_test, tensor_Y_test)

    # 创建 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


    # 实例化模型
    model = EEGNet(batch_size=64, num_class=2).to(device)  # Update parameters as needed
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            data = data.unsqueeze(2)  # 增加维度
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # 计算并打印平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # 评估模型
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 移动数据和标签到 GPU
            data = data.unsqueeze(2)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            y_true.extend(target.tolist())
            y_pred.extend(predicted.tolist())
            y_score.extend(output.softmax(dim=1)[:, 1].tolist())  # 提取正类的概率

    # 计算指标
    acc, auc, ba, recall, precision, f1 = calculate_metrics(np.array(y_true), np.array(y_pred), np.array(y_score))
    # 将结果添加到 DataFrame
    results_df = results_df._append({
        'Subject': subject,
        'Accuracy': acc,
        'AUC': auc,
        'Balanced Accuracy': ba,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1
    }, ignore_index=True)

    print(f'Accuracy: {acc}, AUC: {auc}, BA: {ba}, Recall: {recall}, Precision: {precision}, F1: {f1}')
# 将结果保存为 Excel 文件
results_df.to_excel('results.xlsx', index=False)