import torch
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn import metrics as ms  # 统计库

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c, graph):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.graph_data = graph

        self.F = F.softmax

        self.W = nn.Linear(in_c, out_c, bias=False)  # y = W * x
        self.b = nn.Parameter(torch.Tensor(out_c))

        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

    def forward(self, inputs, graph):
        """
        :param inputs: input features, [B, N, C].
        :param graph: graph structure, [N, N].
        :return:
            output features, [B, N, D].
        """

        h = self.W(inputs)  # [B, N, D]
        outputs = torch.bmm(h, h.transpose(1, 2)) * graph.unsqueeze(0)  # [B, N, D]*[B, D, N]->[B, N, N]      x(i)^T * x(j)

        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))

        attention = self.F(outputs, dim=2)   # [B, N, N]
        return torch.bmm(attention, h) + self.b  # [B, N, N] * [B, N, D]


class GATSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads, graph):
        super(GATSubNet, self).__init__()
        self.graph = graph
        self.attention_module = nn.ModuleList([GraphAttentionLayer(in_c, hid_c, self.graph) for _ in range(n_heads)])
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c, self.graph)

        self.act = nn.LeakyReLU()

    def forward(self, inputs):
        """
        :param inputs: [B, N, C]
        :param graph: [N, N]
        :return:
        """
        outputs = torch.cat([attn(inputs, self.graph) for attn in self.attention_module], dim=-1)  # [B, N, hid_c * h_head]
        outputs = self.act(outputs)

        outputs = self.out_att(outputs, self.graph)

        return self.act(outputs)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x





class DNNNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads, graph1, graph2, n_classes=9):
        super(DNNNet, self).__init__()
        self.graph = graph
        #self.subnet1 = GATSubNet(in_c, hid_c, out_c, n_heads, graph1)
        self.linear1 = nn.Linear(12, n_classes)
        self.linear2 = nn.Linear(10, n_classes)
        #self.subnet2 = GATSubNet(1, 8, 1, 2, graph2)
        #self.selfattention = SelfAttention(dim=1)
        self.linear3 = nn.Linear(22, n_classes)
        self.mlp1 = nn.Linear(106, 48)
        self.mlp2 = nn.Linear(48, 32)
        self.mlp3 = nn.Linear(106, n_classes)
        self.sigmod = nn.Sigmoid()
        # self.subnet = [GATSubNet(...) for _ in range(T)]

    def forward(self, data1, data2, device):

        flow1 = data1  # [B, N, T, C]
        B, N = flow1.size(0), flow1.size(1)
        flow1 = flow1.view(B, -1)
        flow1 = flow1.to(device)
        # #data1操作
        # B, N = flow.size(0), flow.size(1)
        # flow = flow.view(B, N, -1)  # [B, N, T * C]
        # prediction1 = self.subnet1(flow).unsqueeze(2)  # [B, N, 1, C]
        # prediction1 = prediction1.view(prediction1.size(0), -1)
        # # prediction = self.linear(prediction)
        #
        # # data2操作
        flow2 = data2
        # print(flow2.size())
        B, N = flow2.size(0), flow2.size(1)
        flow2 = flow2.view(B, -1)
        flow2 = flow2.to(device)

        flow = torch.cat([flow1, flow2], dim=1)

        #
        # B, N = flow2.size(0), flow2.size(1)
        # flow2 = flow2.view(B, N, -1)  # [B, N, T * C]
        # prediction2 = self.selfattention(flow2)
        # # prediction2 = self.subnet2(flow2).unsqueeze(2)  # [B, N, 1, C]
        # prediction2 = prediction2.view(prediction2.size(0), -1)
        # # prediction2 = flow2.view(flow2.size(0), -1)
        #
        # # res = self.linear1(prediction1) +  1.0 / 11.0 *self.linear2(prediction2)
        #
        # 就这
        # flow = F.relu(self.mlp1(flow))
        # flow = F.relu(self.mlp2(flow))
        flow = self.mlp3(flow)



        # prediction3 = torch.cat([prediction1, prediction2], dim=1)
        # result = self.linear3(prediction3)

        return flow

#自定义交叉熵损失
class MyCrossEntropy(nn.Module):
    def  __init__(self):
        super(MyCrossEntropy, self).__init__()

    def forward(self, x, y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        loss = y * torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss

def normalize(X):
    # 归一化
    X_nor = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1)
    return X_nor


if __name__ == '__main__':
    # graph = np.zeros([12,12])
    #A

    graph = np.array([[0,1,1,1,0,0,0,0,1,0,0,0],
                      [1,0,1,1,1,0,0,0,0,1,0,0],
                      [1,1,0,1,1,1,0,0,0,0,1,0],
                      [0,1,1,0,1,1,1,0,0,0,0,1],
                      [0,0,1,1,0,1,1,1,0,0,1,0],
                      [0,0,0,1,1,0,1,1,0,1,0,0],
                      [0,0,0,0,1,1,0,1,1,0,0,0],
                      [0,0,0,0,0,1,1,0,0,1,0,0],
                      [1,0,0,0,0,0,0,0,0,1,0,1],
                      [0,1,1,0,1,0,0,0,1,0,1,0],
                      [0,0,1,1,0,1,0,0,0,1,0,1],
                      [1,0,1,0,1,1,0,0,1,0,1,0]])

    graph2 = np.array([[0,1,1,1,0,0,0,0,1,0],
                      [1,0,1,1,1,0,0,0,0,1],
                      [1,1,0,1,1,1,0,0,0,0],
                      [0,1,1,0,1,1,1,0,0,0],
                      [0,0,1,1,0,1,1,1,0,0],
                      [0,0,0,1,1,0,1,1,0,1],
                      [0,0,0,0,1,1,0,1,1,0],
                      [0,0,0,0,0,1,1,0,0,1],
                      [1,0,0,0,0,0,0,0,0,1],
                      [0,1,1,0,1,0,0,0,1,0]])


    graph[np.eye(12,dtype=bool)] = 0
    graph = torch.from_numpy(np.float32(graph))
    graph2[np.eye(10,dtype=bool)] = 0
    graph2 = torch.from_numpy(np.float32(graph2))
    #导入数据
    X_r = np.load('PeMS_04/pred_nnet_r.npy')
    X_t = np.load('PeMS_04/pred_nnet_t.npy')
    X_y = np.load('PeMS_04/pred_nnet_v.npy')

    #新加入数据
    x_r_combined = np.load("PeMS_04/man_features_r.npy")
    x_t_combined = np.load("PeMS_04/man_features_t.npy")
    x_v_combined = np.load("PeMS_04/man_features_v.npy")

    #清除NAN和空格
    x_r_combined[np.isnan(x_r_combined)] = 0
    x_t_combined[np.isnan(x_t_combined)] = 0
    x_v_combined[np.isnan(x_v_combined)] = 0

    #归一化
    x_r_combined = normalize(x_r_combined[:,:10])
    x_t_combined = normalize(x_t_combined[:,:10])
    x_v_combined = normalize(x_v_combined[:,:10])

    #添加空列并转换
    x_r_combined = x_r_combined.reshape(x_r_combined.shape[0],10,1,1)
    x_t_combined = x_t_combined.reshape(x_t_combined.shape[0],10,1,1)
    x_v_combined = x_v_combined.reshape(x_v_combined.shape[0],10,1,1)
    #导入数据
    label_r = np.load('PeMS_04/lb_r.npy')   # 0 ,1 ,2, 3,4 ....
    label_t = np.load('PeMS_04/lb_t.npy')  # 0 ,1 ,2, 3,4 ....
    label_v = np.load('PeMS_04/lb_v.npy')  # 0 ,1 ,2, 3,4 ....
    print(X_r.shape)
    train_x = X_r.reshape(8388,12,8,1)

    # print(train_x.shape)
    # ohe = OneHotEncoder()
    # ohe.fit(label_r.reshape(-1, 1))
    # train_y = ohe.transform(label_r.reshape(-1, 1)).toarray()
    # train_y = torch.from_numpy(np.float32(train_y))
    train_y = torch.from_numpy(label_r)

    # train_y = label_r.reshape(8388,12,1,1)
    print(train_y.shape)
    train_x = torch.from_numpy(np.float32(train_x))
    x_r_combined = torch.from_numpy(np.float32(x_r_combined))

    batch_size = 64
    epochs = 10
    model = DNNNet(in_c = 8, hid_c = 4, out_c = 1,n_heads = 2, graph1 = graph , graph2 = graph2)
    # print(model)
    device = torch.device("cpu")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),lr= 2e-5)

    # 开始训练
    torch_dataset = Data.TensorDataset(train_x, x_r_combined, train_y)  # 得到一个元组(x, y)
    # torch_dataset = Data.TensorDataset(train_x, train_y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,  # 每次训练打乱数据， 默认为False
        num_workers=2,  # 使用多进行程读取数据， 默认0，为不使用多进程
    )

    #自定义交叉熵损失函数
    # crossEntropy = MyCrossEntropy()
    crossEntropy = nn.CrossEntropyLoss()
    #循环
    for epoch in range(epochs):
        for step, (batch_x1, batch_x2, batch_y) in enumerate(loader):
            # 1.8000  优化1次
            # 2.1     优化8000次
            # 3.64    8000/64次
            prd = model(batch_x1, batch_x2, device)
            loss = crossEntropy(prd,batch_y.long())
            # loss2 = loss_function2(prd,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == 100:
                print(epoch, "loss:", loss.data)

    X_t = np.load('PeMS_04/pred_nnet_t.npy')
    test_x = X_t.reshape(-1, 12, 8, 1)
    test_x = torch.from_numpy(np.float32(test_x))
    #测试部分
    label_t = np.load('PeMS_04/lb_t.npy')
    ohe = OneHotEncoder()
    ohe.fit(label_t.reshape(-1, 1))
    test_y = ohe.transform(label_t.reshape(-1, 1)).toarray()
    test_y = torch.from_numpy(np.float32(test_y))

    y_pred = model(test_x, torch.from_numpy(np.float32(x_t_combined)), device)
    # print((test_y.argmax(axis=1) == y_pred.argmax(axis=1)).sum()/len(test_y))
    # print((test_y.argmax(axis=1) == y_pred.argmax(axis=1)).sum() / len(test_y))
    #print(y_pred.sum(axis=0))

    print((torch.from_numpy(label_t) == y_pred.argmax(axis=1)).sum() / len(test_y))
    y_pred = y_pred.argmax(axis=1)
    test_y = test_y.argmax(axis=1)

    print("准确率为:{0:%}".format(ms.accuracy_score(test_y, y_pred)))
    print("精确率为:{0:%}".format(ms.precision_score(test_y, y_pred, average='macro')))
    print("召回率为:{0:%}".format(ms.recall_score(test_y, y_pred, average='macro')))
    print("F1分数为:{0:%}".format(ms.f1_score(test_y, y_pred, average='macro')))
    print("Fbeta为:{0:%}".format(ms.fbeta_score(test_y, y_pred, beta=1.2, average='macro')))

    # 训练
    #prd = model(train_x, x_r_combined, device)
    #print(prd.shape)






