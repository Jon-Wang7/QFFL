import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np
import math


class QFNN(nn.Module):
    def __init__(self):
        super(QFNN, self).__init__()

        # 经典神经网络层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)  # 修改为16维，匹配量子电路输入

        # 量子电路参数
        self.n_qubits = 4  # 4个量子比特，需要16维输入
        self.n_layers = 2

        # 初始化量子设备
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # 量子电路权重
        self.q_weights = nn.Parameter(torch.randn(self.n_layers, self.n_qubits, 3))

        # 最终分类层
        self.fc4 = nn.Linear(16 + self.n_qubits, 10)  # 修改输入维度

        # 量子电路
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # 振幅编码
            qml.AmplitudeEmbedding(inputs, wires=range(self.n_qubits), normalize=True)

            # 变分层
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.Rot(*weights[layer][qubit], wires=qubit)
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

            # 测量
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.quantum_circuit = quantum_circuit

    def forward(self, x):
        # 经典神经网络前向传播
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # 输出16维

        # 量子处理
        batch_size = x.shape[0]
        quantum_output = torch.zeros((batch_size, self.n_qubits), device=x.device)

        for i in range(batch_size):
            # 提取并预处理输入
            sample = x[i]
            # 归一化
            sample = F.normalize(sample, p=2, dim=0)

            # 执行量子电路并转换为张量
            quantum_output[i] = torch.tensor(self.quantum_circuit(sample, self.q_weights), device=x.device)

        # 将量子输出与经典特征结合
        x = torch.cat([x, quantum_output], dim=1)  # 16 + 4 = 20维

        # 最终分类
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

