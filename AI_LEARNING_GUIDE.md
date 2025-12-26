# 🧩 Java 后端开发者的 AI 入门指南：以四人围棋为例

你好！作为一名资深的 Java 后端开发，你可能习惯了确定性的逻辑（if-else, try-catch, 事务控制）。欢迎来到机器学习的世界，这里更像是一个**高度自动化的统计学系统**。

本指南将通过你熟悉的后端架构视角，带你拆解这个四人围棋 AI 的实现原理。

---

## 1. 核心工具栈：ML 界的“Spring Boot”

在本项目中，我们使用了以下工具，你可以对照 Java 生态来理解：

| 工具 | ML 角色 | Java 类比 | 作用 |
| :--- | :--- | :--- | :--- |
| **PyTorch** | 深度学习框架 | **Spring Boot / JDK** | 提供基础的张量运算、神经网络层、反向传播引擎。 |
| **ONNX** | 模型交换格式 | **JSON / Protobuf** | 跨语言的模型序列化协议。Python 训练完，导出为 ONNX，前端 JS 就能直接读。 |
| **ONNX Runtime** | 推理引擎 | **JVM / Docker** | 负责运行 `.onnx` 文件的环境，在浏览器或服务端高效执行模型预测。 |
| **MCTS** | 搜索算法 | **分布式任务编排 / 模拟引擎** | 负责“思考”和“规划”，是 AI 智力的放大器。 |

---

## 2. 核心架构：AlphaZero 的“双循环”

AlphaZero 的精髓在于两个循环的相互博弈。

### A. 业务员循环：自我博弈 (Self-Play)
*   **角色**：[train_parallel.py](file:///Users/chengshang/Code/trae_projects/4/4-player-go/training/train_parallel.py) 里的 `Worker`。
*   **职责**：产生数据。
*   **过程**：
    1.  AI 自己跟自己下棋。
    2.  每一步都由 **MCTS (逻辑推理)** + **Neural Network (直觉预判)** 共同完成。
    3.  **记忆存储**：把每一个局面 (State)、MCTS 的建议 (Policy)、最终的胜负 (Value) 存入数据库（内存列表）。

### B. 总部循环：模型训练 (Training)
*   **角色**：主进程的 `optimizer.step()`。
*   **职责**：优化模型参数（修正 Bug）。
*   **过程**：
    1.  从数据库读取成千上万条“记忆”。
    2.  **算账 (Calculate Loss)**：比较“当时 AI 的直觉”和“最终真实的结局”。
    3.  **反向传播 (Backprop)**：通过数学方式，精准定位到是神经网络哪一层、哪个参数导致了判断误差，并微调它。

---

## 3. 深入浅出：MCTS 到底在干什么？

你可以把 **MCTS (蒙特卡洛树搜索)** 想象成一个**高性能的分布式推演引擎**。

1.  **Selection (选择)**：从当前局面开始，根据“直觉”挑出几条最有希望的路径。
2.  **Expansion (扩张)**：在选中的路径上往前多看一步。
3.  **Evaluation (评估)**：**最核心！** 它不往下下完了，而是直接调神经网络的接口：“大哥，你看看这个新局面，估个分。”
4.  **Backpropagation (回传)**：把这个估分顺着树传回去，更新路径上的平均得分。

**为什么它能“纠偏”？**
即便神经网络说 A 点好，但 MCTS 模拟后发现 A 点后面全是坑，那么 MCTS 最终给出的建议就会避开 A 点。这种“深思熟虑”后的结果，就是神经网络下一次要学习的“正确答案”。

---

## 4. 神经网络：它不是查表，是“特征识别”

为什么 AI 没见过的局面也能下？

*   **卷积层 (Convolutional Layers)**：它像是一个个“滤镜”。有的滤镜专门看“这里有没有三个子连在一起”，有的看“这里是不是死棋”。
*   **泛化 (Generalization)**：神经网络学习的是这些**局部特征的组合**。就像你没见过某款新车，但你认得轮子、方向盘和车灯，所以你依然知道它是一辆车。

---

## 5. 给 Java 开发者的实战建议

### 如何部署到前端？
1.  **Export**：运行 [export_onnx.py](file:///Users/chengshang/Code/trae_projects/4/4-player-go/training/export_onnx.py)，将 `.pth` (PyTorch 私有格式) 转为 `.onnx` (标准协议)。
2.  **Load**：在 [NeuralAI.ts](file:///Users/chengshang/Code/trae_projects/4/4-player-go/src/game/NeuralAI.ts) 中，通过 `ort.InferenceSession.create('/models/model.onnx')` 加载。
3.  **Predict**：将棋盘数组转为 `Tensor`，丢给模型，拿到 Policy（落子概率）和 Value（胜率预估）。

### 训练挂了怎么办？
*   **Loss 不降反升**：通常是学习率 (Learning Rate) 太大，就像线程池满了导致拒绝策略，需要调小。
*   **模型不走棋**：检查 `get_valid_moves` 逻辑。如果规则判定全是无效步，AI 就“瘫痪”了。

---

## 6. 总结

AI 训练的过程，其实就是**把“昂贵的逻辑推理 (MCTS)”转化为“廉价的直觉判断 (Neural Network)”**的过程。

作为 Java 开发者，你可以把神经网络看作是一个**极其复杂的 `Function<Board, Result>`**，而我们的所有训练工作，都是在通过“单元测试（胜负规则）”和“大数据分析（自我博弈）”来自动生成这个函数的内部实现。

希望这份文档能帮你开启 AI 的大门！如果有任何具体的代码细节想深挖，随时问我。
