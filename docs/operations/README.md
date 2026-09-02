# LightRL 通用运维约定

这里放不依赖具体节点、RJob 名称或临时 IP 的可复用运维文档。带有现场地址、租约
ID、代理拓扑和一次性故障结论的内容应留在本地 `docs/records/operations/`。

- [checkpoint-wandb.md](checkpoint-wandb.md)：checkpoint、W&B offline 和运行目录的
  持久化约定。
- Docker worker 的部署与站点恢复记录暂保留在 `docs/records/operations/worker/`，
  待抽象出不含站点参数的版本后再提升到此目录。
