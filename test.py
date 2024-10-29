import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
matplotlib.rc("font", family='YouYuan')


# 定义多标签MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc3(self.relu(out))
        out = self.sigmoid(out)
        return out


def main():
    X = get_x_data()
    Y = get_y_data()
    # 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    print(Y_test.shape)
    X_train, X_test, Y_train, Y_test = torch.FloatTensor(X), torch.FloatTensor(X), torch.LongTensor(
        Y), torch.LongTensor(Y)
    print(len(X_train))
    print(len(Y_train))
    print(Y_test.shape)
    # 定义模型超参数
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = 7
    learning_rate = 0.01
    num_epochs = 1000

    # 初始化模型、损失函数和优化器
    model = MLP(input_size, hidden_size, output_size)
    # criterion = nn.BCELoss()  # 使用二元交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True)

    # 训练模型
    losses = []  # 用于保存每个epoch的损失值
    accuracies = []  # 用于保存每个epoch的准确度
    for epoch in range(num_epochs):
        epoch_loss = 0.0  # 用于保存当前epoch的总损失
        correct_predictions = 0  # 用于保存当前epoch的正确预测数量
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)  # 累积当前batch的损失
            # 计算当前batch的准确度
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            total += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # 计算平均损失和准确度
        epoch_loss /= len(train_loader.dataset)
        accuracy = correct_predictions / total

        losses.append(epoch_loss)
        accuracies.append(accuracy)
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

    # 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        print(predicted)
        print(Y_test)
        accuracy = (predicted == Y_test).sum().item() / 20
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
    # 绘制损失趋势图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()

    # 绘制准确度趋势图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Time')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    main()
