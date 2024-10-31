import numpy as np

def preprocess_data(data, flatten_mode="flatten"):
    """
    根据 flatten_mode 参数选择不同的数据展平或矩阵保留方式。
    参数:
        data: 输入的原始数据 (28, 4, 7) 格式
        flatten_mode: 选择展平方式的字符串
                      "flatten" - 完全展平为 1D
                      "28x4x7" - 保持原始三维格式
                      "7x(4x28)" - 7 行，每行包含 4x28 特征的 2D 矩阵
                      "4x(7x28)" - 4 行，每行包含 7x28 特征的 2D 矩阵
    返回值:
        处理后的数据
    """
    if flatten_mode == "flatten":
        # 完全展平
        return data.flatten()  # 1D array, shape (784,)
    elif flatten_mode == "28x4x7":
        # 保持原始三维结构
        return data  # 3D array, shape (28, 4, 7)
    elif flatten_mode == "7x(4x28)":
        # 7 个关节点，每个关节一个 (4*28) 的特征向量
        return data.reshape(7, 4 * 28)  # 2D array, shape (7, 112)  
    elif flatten_mode == "4x(7x28)":
        # 4 个特征，每个特征是一个 7x28 的矩阵
        return data.reshape(4, 7 * 28)  # 2D array, shape (4, 196)
    else:
        raise ValueError("Invalid flatten_mode. Choose from 'flatten', '28x4x7', '7x(4x28)', '4x(7x28)'.")


class RBFNetwork:
    def __init__(self,num_centers,num_classes,flatten_mode = "flatten"):
        self.num_centers = num_centers
        self.num_classes = num_classes
        self.flatten_mode = flatten_mode
        self.num_features = None
        self.centers = None
        self.sigmas = np.ones(num_centers)  # 初始化标准差
        self.weights = None


    def preprocess_input(self, data):
        processed_data = preprocess_data(data, flatten_mode=self.flatten_mode)
        self.num_features = processed_data.size if self.num_features is None else self.num_features
        return processed_data


    def rbf_function(self, x, center, sigma):
        return np.exp(-np.linalg.norm(x - center)**2 / (2 * sigma**2))

    def calculate_hidden_layer(self, X):
        hidden_layer = np.zeros((X.shape[0], self.num_centers))
        for i, sample in enumerate(X):
            for j, center in enumerate(self.centers):
                hidden_layer[i, j] = self.rbf_function(sample, center, self.sigmas[j])
        return hidden_layer

    def predict(self, X):
        X_processed = np.array([self.preprocess_input(sample) for sample in X])
        hidden_layer = self.calculate_hidden_layer(X_processed)
        scores = np.dot(hidden_layer, self.weights)
        return (scores > 0).astype(int) # 将scores的元素与 0 比较，并将结果转换为整数格式（0或1）

    def train(self, X, y, lr=0.01, epochs=100):
        X_processed = np.array([self.preprocess_input(sample) for sample in X])
        if self.centers is None:
            self.centers = np.random.randn(self.num_centers, self.num_features)
        if self.weights is None:
            self.weights = np.random.randn(self.num_centers, self.num_classes)

        for epoch in range(epochs):
            hidden_layer = self.calculate_hidden_layer(X_processed)
            output = np.dot(hidden_layer, self.weights)
            error = y - output
            self.weights += lr * np.dot(hidden_layer.T, error)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Error: {np.mean(error ** 2)}")

    