# 要点

这次我们用同样的迷宫例子来实现 RL 中另一种和 Qlearning 类似的算法, 叫做 Sarsa (state-action-reward-state_-action_). 我们从这一个简称可以了解到, Sarsa 的整个循环都将是在一个路径上, 也就是 on-policy, 下一个 state_, 和下一个 action_ 将会变成他真正采取的 action 和 state. 和 Qlearning 的不同之处就在这. Qlearning 的下个一个 state_ action_ 在算法更新的时候都还是不确定的 (off-policy). 而 Sarsa 的 state_, action_ 在这次算法更新的时候已经确定好了 (on-policy).

[![IMAGE ALT TEXT](https://morvanzhou.github.io/static/results/ML-intro/q5.png)](https://morvanzhou.github.io/static/results/reinforcement-learning/maze%20sarsa.mp4)

# 算法

![img](https://morvanzhou.github.io/static/results/reinforcement-learning/3-1-1.png)

他在当前 state 已经想好了 state 对应的 action, 而且想好了 下一个 state_ 和下一个 action_ (Qlearning 还没有想好下一个 action_)
更新 Q(s,a) 的时候基于的是下一个 Q(s_, a_) (Qlearning 是基于 maxQ(s_))
这种不同之处使得 Sarsa 相对于 Qlearning, 更加的胆小. 因为 Qlearning 永远都是想着 maxQ 最大化, 因为这个 maxQ 而变得贪婪, 不考虑其他非 maxQ 的结果. 我们可以理解成 Qlearning 是一种贪婪, 大胆, 勇敢的算法, 对于错误, 死亡并不在乎. 而 Sarsa 是一种保守的算法, 他在乎每一步决策, 对于错误和死亡比较铭感. 这一点我们会在可视化的部分看出他们的不同. 两种算法都有他们的好处, 比如在实际中, 你比较在乎机器的损害, 用一种保守的算法, 在训练时就能减少损坏的次数.

# 算法的代码形式

首先我们先 import 两个模块, maze_env 是我们的环境模块, 已经编写好了, 大家可以直接在这里下载, maze_env 模块我们可以不深入研究, 如果你对编辑环境感兴趣, 可以去看看如何使用 python 自带的简单 GUI 模块 tkinter 来编写虚拟环境. 我也有对应的教程. maze_env 就是用 tkinter 编写的. 而 RL_brain 这个模块是 RL 的大脑部分, 我们下节会讲.

```
from maze_env import Maze
from RL_brain import SarsaTable
```

下面的代码, 我们可以根据上面的图片中的算法对应起来, 这就是整个 Sarsa 最重要的迭代更新部分啦.

```
def update():
    for episode in range(100):
        # 初始化环境
        observation = env.reset()

        # Sarsa 根据 state 观测选择行为
        action = RL.choose_action(str(observation))

        while True:
            # 刷新环境
            env.render()

            # 在环境中采取行为, 获得下一个 state_ (obervation_), reward, 和是否终止
            observation_, reward, done = env.step(action)

            # 根据下一个 state (obervation_) 选取下一个 action_
            action_ = RL.choose_action(str(observation_))

            # 从 (s, a, r, s, a) 中学习, 更新 Q_tabel 的参数 ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 将下一个当成下一步的 state (observation) and action
            observation = observation_
            action = action_

            # 终止时跳出循环
            if done:
                break

    # 大循环完毕
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
```

下一节我们会来讲解 SarsaTable 这种算法具体要怎么编.

如果想一次性看到全部代码, 请看这里：

https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/3_Sarsa_maze
