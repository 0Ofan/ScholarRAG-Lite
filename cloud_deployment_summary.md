# 完整可下载项目与 Community Cloud 适配

## 已实现

- 保留两种完整 12,000 篇检索方法及原始 BGE、FAISS IndexFlatIP、论文 max-chunk 聚合、V1/V2 和 Top-3 默认。
- 新增 `community/app.py`，既可供新用户本地运行，也作为 Streamlit Community Cloud 的入口。
- 新增 CPU 依赖清单与 Linux libgomp1 声明，未修改现有 Windows .venv 或重新安装依赖。
- GitHub 仓库保存源码、小型配置和 Dashboard 结果；大文件使用同一仓库 `runtime-v1` Release 的五个分卷。
- `deployment/cloud_manifest.json` 固定分卷和 19 个原始运行文件的 SHA-256；下载器逐卷校验，解包时逐文件校验，并拒绝覆盖不同内容的现有文件。
- 下载完整后在 Streamlit 所在机器内使用这些材料；不访问作者电脑。损坏下载不会进入检索流程。
- 云端按位置读取原始 JSONL，常驻一个 CPU 模型与一个原始索引；切换方法重新加载索引，并发检索使用共享锁。
- 原始 root app.py 入口、Prompt、检索算法和历史数据未修改；通过 `SCHOLARRAG_CLOUD` 控制加载策略。
- README 和 `community/README.md` 说明新用户安装、运行、Release 发布和云端配置。最终报告/PPT不纳入 Git。

## 验证

- 15 项新增/直接相关单元与回归测试通过，涉及按需原文读取、检索顺序与分数、引用来源、单模型缓存、索引切换、下载校验、禁止覆盖、路径检查及原 Prompt 结构。
- 7 个新增/受影响 Python 文件语法检查通过。
- 完整 19 个原始运行文件已在本地依据 manifest 校验，下载器识别已有完整材料，无重复下载。
- 单题 Q07 本地 CPU 检查按 Fixed → Sentence-aware → Fixed 切换；两方法各有 12,000 篇、693,724 / 706,366 个片段。
- Fixed Top-10 论文 ID 和前三条证据 Chunk ID 与 Phase 3.6 保存记录一致；全文和来源对应检查通过。
- 包含实际 Streamlit/图表模块导入的 Windows CPU 检查峰值 RSS 约 1.976 GiB，模型加载 1 次、索引加载 3 次。逐次总耗时约 1.276 / 1.159 / 1.220 秒，含 JSONL 位置扫描、索引加载及检索，不含首次 Python 导入、网络下载、浏览器显示或 LLM。
- 原始机器可用 CUDA 版 PyTorch，但本次显式在 CPU 运行。这不是 Linux CPU wheel 或 Community Cloud 的内存保证。
- 本地云端入口启动后 `/_stcore/health` 返回 HTTP 200 `ok`，检查后已停止服务。
- 真实浏览器交互与云端依赖安装尚未验证。一项旧 AppTest 在沙箱内未结束，已停止，不计入通过数量；采用直接相关测试和本地服务健康检查。
- 未调用 DeepSeek，未重新生成 Embedding/索引，未运行 20 题实验。

## 交付与尚待操作

- 用户已将仓库设置为公开；只读 API 核实 `private=false`，未由 Codex 修改可见性。
- 新源码已提交并推送，功能提交为 `c958661`；后续文档提交记录发布状态。
- runtime-v1 的五个分卷已发布：https://github.com/0Ofan/ScholarRAG-Lite/releases/tag/runtime-v1 。GitHub 返回的五个 SHA-256 与清单全部一致。
- 未带凭据的公开 Release 元数据读取成功，首个分卷匿名范围下载返回 HTTP 206；下载的 1 MiB 字节与本地原文件一致。未将该范围检查描述为完整远程冷下载。
- 新增发布工具复用 Git 已保存的凭据，仅在进程内使用，不保存或输出凭据。实际代理 TLS 握手不稳定时只对发送请求前的连接失败进行有限重试。
- 下一步在 share.streamlit.io 选择 main / community/app.py / Python 3.13 创建应用。
- DeepSeek 配置填写平台 Secrets，真实密钥不进入项目提交。
- 尚无真实云端链接。Community Cloud 构建、运行和浏览器验收仍未完成；不能以 GitHub 上传成功代替云端验收。
- 已有离线研究模块源码保留，但完整运行包不包含所有历史索引、探索模型及问答记录；其重现实验需求在文档中单独说明，不宣称全历史实验下载即重跑。

机器可读本地测量：`experiments/results/cloud_deployment/cpu_smoke.json`。
