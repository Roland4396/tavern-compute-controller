# tavern-compute-controller

GPU 放置调度、本地文本适配和 H3 任务幂等、取消、状态交接。

同一控制器进程内按 scheduler、local_text_wire、h3_jobs、h3_state 和 h3_compute 分模块，避免拆散事务边界。自行把节点清单挂载到 /app/nodes.json，并在私有环境配置 SSH 与回调。生产节点、端口映射、锁、运行态数据库均不随包下发。

模板 nodes.example.json 仅用于演示字段，没有实际地址。镜像里使用 runner 账户，远端示例用户 worker；配置时用自己的主机名、用户和路径。示例不是已配置的真实节点。
