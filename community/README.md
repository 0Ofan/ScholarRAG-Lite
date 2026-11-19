# 完整项目交付与 Streamlit Community Cloud

## 交付结构

- GitHub 源码仓库包含应用、核心/离线实验模块、Prompts、测试、实验配置、两份 Dashboard 历史结果以及运行说明。
- 同一仓库的 `runtime-v1` Release 存放完整 12,000 篇两种方法的运行材料，总压缩大小约 2.32 GiB。
- 仓库中 `deployment/cloud_manifest.json` 固定各文件及各下载分卷的 SHA-256。
- 别人下载源码后执行 README 启动命令，自动从 GitHub 下载材料；无须访问作者电脑、重建索引或生成 Embedding。
- 完整应用不等于全部历史实验归档：Pilot/Medium 索引、原始全文全集、Reranker 等探索模型与历史回答不在运行包中。它们不是三个页面运行所需文件；依赖这些历史材料的旧实验测试不能在新下载后全量执行。可独立运行本次的下载、存储和检索单元测试。
- 报告/PPT、密钥、.venv 和 .cache 不纳入软件提交。

## 1 上传更新后的源码

当前源码与 runtime-v1 Release 已发布，首次使用者直接按根目录 README 运行，或继续第 3 节创建云端应用。下列命令保留为维护者更新时的参考。

在本机终端进入项目根目录后：

```powershell
git add README.md .gitignore src/cloud_retriever.py src/runtime_download.py src/rag_service.py community deployment/prepare_release.py deployment/verify_cloud.py deployment/cloud_manifest.json .env.example tests/test_cloud_deployment.py cloud_deployment_summary.md
git commit -m "Support complete downloadable project and Streamlit Community Cloud"
git -c http.proxy=http://127.0.0.1:7890 push origin main
```

代理端口沿用用户已验证的 7890；未使用代理的网络可以省略 `-c http.proxy=...`。
只暂存上面明确列出的文件，保留其他历史文档和结果。

## 2 发布 GitHub Runtime Release

已发布地址：https://github.com/0Ofan/ScholarRAG-Lite/releases/tag/runtime-v1 。五个分卷的 GitHub SHA-256 与固定清单一致，并完成匿名范围下载核对。无需再次发布或重复上传。

打开仓库 **Releases → Create a new release**：

1. 创建标签 `runtime-v1`，目标分支 `main`。
2. 标题可写 `Complete 12000-paper runtime`。
3. 附加 `deployment/dist/github-release/` 内以下五个文件：
   `runtime.part01`、`runtime.part02`、`runtime.part03`、`runtime.part04`、`runtime.part05`。
4. 等上传全部结束，发布 Release。不要上传旧的整个 2.32 GiB TAR 文件，它超过单个 Release 资产的 2 GiB 限制。

这些是二进制数据分卷，不需要用户手动合并。应用按顺序下载、校验并解包。
Release 必须能从未登录浏览器下载；如果仓库是私有的，目前此下载入口无法使用，需先确定共享方式，不能声称已完成他人可下载交付。
原始论文及第三方模型保留其各自许可，软件仓库不改变原始材料的权利归属。

维护者后续需要发布时可使用 `python -m deployment.publish_release`；该工具只从 Git 凭据管理器读取已登录的 GitHub 凭据，在进程内通过 stdin 传给 curl，不把凭据写入文件或参数。已存在且校验一致的资产会跳过，不同资产不会被覆盖。

## 3 在 share.streamlit.io 部署

1. 用 GitHub 登录 Streamlit Community Cloud，选择 **Create app / Deploy an app**。
2. Repository：`0Ofan/ScholarRAG-Lite`。
3. Branch：`main`。
4. Main file path：`community/app.py`，不是根目录 `app.py`。
5. Advanced settings 中选择 **Python 3.13**。
6. 在 Secrets 填入以下 TOML 配置，密钥自行填写，不发到对话或仓库：

```toml
DEEPSEEK_API_KEY = "填写实际密钥"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_MAX_TOKENS = 512
DEEPSEEK_TEMPERATURE = 0.3
DEEPSEEK_TIMEOUT = 30
```

7. 点击 Deploy，等待安装 CPU 依赖、下载和校验完整运行包。第一次启动需要下载数 GiB 文件，后续查询不会再次下载。
8. 成功后复制平台分配的 `https://…streamlit.app` 链接给组员。仓库 README 仅在地址真实可用后补入链接。

依赖清单在入口同目录的 `community/requirements.txt`，平台会优先使用它；CPU PyTorch 不需要 CUDA。
`community/packages.txt` 提供 Linux 的 libgomp1。
重启、休眠恢复可能触发重新准备数据，不应将下载时间当成热查询时间。

## 4 验收

- Search & Answer：默认 Large Fixed、Top-3、V1；提交问题后能检索并生成有来源的回答。
- 切换 Sentence-aware：仍是完整 12,000 篇，切换方法时会重新加载索引。
- Method Comparison：两种结果均可显示，不调用 LLM。
- Dashboard：读取历史对比表，浏览页面不重跑实验。
- V2：Experimental，保留来源角色和核查提示；Unverified 不代表医学事实认证。
- API 失败仍保留本次检索证据；并发访问需要实际验收，当前检索使用共享锁控制内存。

本地 Windows CPU 烟雾检查不能替代 Community Cloud 的内存、依赖安装、网络下载及浏览器验收。
云端资源不足时先看平台日志，不能声称已成功部署，也不自动缩小语料或改检索算法。

## 本地可重复检查

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_cloud_deployment tests.test_deployment -v
.\.venv\Scripts\python.exe -m src.runtime_download
```

后一个命令从已发布 Release 获取并校验运行数据；若本地已经有完整相同数据，只校验，不重新下载。
本次单题 CPU 资源与历史 Top-10 对照可用 `python -m deployment.verify_cloud` 运行，但它不是新的 20 题实验，不调用 DeepSeek。
