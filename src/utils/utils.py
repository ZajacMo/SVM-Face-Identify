import numpy as np

class BPNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        # 输出层使用softmax激活函数，适合多分类问题
        logits = np.dot(self.hidden, self.weights2) + self.bias2
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # 数值稳定版softmax
        self.output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return self.output
    
    def backward(self, X, y, output, learning_rate):
        output_error = y - output
        # softmax配合交叉熵损失时，输出层delta直接等于误差
        output_delta = output_error
        
        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * self.sigmoid_deriv(self.hidden)
        
        self.weights2 += self.hidden.T.dot(output_delta) * learning_rate
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights1 += X.T.dot(hidden_delta) * learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
    
    def train(self, X, y, epochs, learning_rate, batch_size=32):
        self.loss_history = []  # 初始化损失历史记录
        self.learning_rate = learning_rate  # 保存学习率参数
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output, learning_rate)
            
            if epoch % 100 == 0:
                # 使用交叉熵损失代替均方误差，更适合分类问题
                output = self.forward(X)
                loss = -np.mean(y * np.log(output + 1e-10))  # 添加微小值防止log(0)
                self.loss_history.append((epoch, loss))  # 记录损失历史
                print(f"Epoch {epoch}, Loss: {loss}")

def preprocess_image(image_path, image_size):
    from PIL import Image
    import numpy as np
    img = Image.open(image_path).convert('RGB')
    img = img.resize(image_size)
    img_array = np.array(img) / 255.0
    return img_array.flatten()