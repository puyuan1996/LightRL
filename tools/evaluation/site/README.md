# 站点/节点相关配置(site profiles)

通用工具层(`core/`)刻意**不**包含以下站点强相关能力,因为它们绑定具体
集群/节点拓扑,无法通用化:

- **brainctl RJob 与 model relay**:把远端正跑的 SGLang 通过 relay 暴露成
  本机 `api_base`(`model_http_relay` 思路),依赖集群提交系统与端口规划。
- **节点级 docker 外部网络覆盖**:`extra_docker_compose` 里的
  `networks.external` 等写法因节点 docker 配置而异。
- **正向代理(如 tinyproxy)**:地址、端口、no_proxy 清单都是站点属性。
- **离线镜像打包/导入链**(docker save/load、wheelhouse 预置):属于部署流程。

## 迁移指引

1. 用本目录的 profile 示例(或自建 YAML)描述站点差异:代理 env、
   compose 网络覆盖、external serving 的 `api_base`(指向 relay)。
2. 若评测必须跑在远端节点:用 `ssh <node> 'bash -s' < script` 或
   `systemd-run` 把 `eval_cli.py run --config <site-profile>.yaml` 包装到
   远端执行;仓库 `runs/` 下遗留的一次性脚本(如多 ckpt 接力回收的
   supervise 脚本)可作为参考,但不要直接复用其硬编码路径。
3. managed serving 在远端节点同样可用:把 profile 里 `serving.mode` 改成
   `managed`,工具层会在远端本机起停 SGLang。
