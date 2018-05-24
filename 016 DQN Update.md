接着上节内容, 我们来定义 DeepQNetwork 的决策和思考部分.

# 代码主结构

定义完上次的神经网络部分以后, 这次我们来定义其他部分. 包括:

```
class DeepQNetwork:
    # 上次的内容
    def _build_net(self):

    # 这次的内容:
    # 初始值
    def __init__(self):

    # 存储记忆
    def store_transition(self, s, a, r, s_):

    # 选行为
    def choose_action(self, observation):

    # 学习
    def learn(self):

    # 看看学习效果 (可选)
    def plot_cost(self):

```
