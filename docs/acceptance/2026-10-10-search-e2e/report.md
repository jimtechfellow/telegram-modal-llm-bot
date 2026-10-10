研究报告

关键发现
Modal通过维护一个由空闲健康GPU组成的缓冲区，允许新副本快速调度至这些单元，并在缓冲区过大或副本关闭时异步释放设备，从而优化冷启动与资源回收。[1][3]
该机制将GPU初始化移出热路径，通过预先运行空闲GPU并异步补充新设备，实现了真正的无服务器GPU体验，支持多应用共享底层硬件资源。[1][3]

局限
资料主要描述架构设计思路，缺乏具体的空闲超时阈值、回收触发条件及实际性能基准数据，细节推断基于摘要片段。

来源（搜索返回，未独立全文核验）
[1] How we achieved truly serverless GPUs
https://modal.com/blog/truly-serverless-gpus
[3] 如何实现真正的无服务器GPU — 实録
https://shilu.ai/articles/modal/blog-truly-serverless-gpus
