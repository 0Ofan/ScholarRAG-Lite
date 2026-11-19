# 云服务器部署

目标仓库：https://github.com/0Ofan/ScholarRAG-Lite.git

本配置用于 Linux CPU 云服务器，保留 Large 两种检索、V1 默认、V2 Experimental 和三个页面。
源码和 runtime-v1 Release 现已公开发布。当前首选的 Community Cloud 部署入口、完整下载运行方式见 `community/README.md`；下方首次 Git 上传内容是早期准备记录，不需要重复执行。
资源起点建议为 4 核 CPU、16 GiB RAM；实际并发能力尚未测量。
不适合把当前完整运行数据直接放入免费 Streamlit Community Cloud。

## 本地准备数据

在项目根目录使用 Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe deployment/package_runtime.py
.\.venv\Scripts\python.exe deployment/package_runtime.py --build
```

输出在 `deployment/dist/`，该目录不加入 Git。
数据包只包括两套 Large 索引、对应 Chunk 原文和 BGE-small 模型快照。
不会打包原始论文全集、Embedding NPY、MiniLM、Reranker、密钥或历史问答。
模型快照以实际文件内容打包，避免 Hugging Face 符号链接在 Linux 上失效。
Chunk 已带完整正文与来源元数据；其 source_path 保留为历史来源标识，本应用不依赖打开原始全文文件。

## 上传代码

首次提交前检查 Git 清单，特别是密钥和超过 100 MB 的文件。不要强制推送覆盖远程历史。
若目标仓库已有提交，应先在独立目录拉取合并；仓库尚未核实前不直接执行 push。
仓库只需代码、Prompts、部署配置及两份已允许加入 Git 的 Dashboard comparison.json。
当前本地 `.venv` 和 `.env` 都不上传。

本次已经初始化本地 `main` 分支、设置 origin 并暂存源代码，尚未创建提交或推送。
由于当前执行环境连接 GitHub 被重置，请在 Windows PowerShell 中先执行：

```powershell
cd D:\VSproject\BD\Project
git diff --cached --stat
git ls-remote --heads origin
```

若第二条 Git 命令成功且没有输出任何分支，说明远程暂无分支，可以执行：

```powershell
git commit -m "Prepare ScholarRAG Lite application and deployment"
git push -u origin main
```

如果提示未配置提交身份，使用项目级设置（邮箱由你自行填写）：

```powershell
git config --local user.name "0Ofan"
git config --local user.email "你的 GitHub 提交邮箱"
```

再重新执行 commit/push。GitHub 登录通过浏览器或 Git 凭据管理器完成，不把 Token 写进远程 URL。
若 `ls-remote` 有分支输出或连接失败，先停止上述提交/推送步骤并保留输出，确认远程历史后再合并。
不要执行 `git push --force`。

## 在服务器运行

前提：服务器已有 Git、Docker Engine 和 Compose。安装系统软件由服务器管理员处理。
在选定的服务器项目目录执行：

```bash
git clone https://github.com/0Ofan/ScholarRAG-Lite.git
cd ScholarRAG-Lite
mkdir -p deployment/runtime
# 将本地两个导出文件通过 scp 等方式传至 deployment/dist/
tar -xzf deployment/dist/scholarrag_runtime.tar.gz -C deployment/runtime
python3 deployment/package_runtime.py --root deployment/runtime --verify deployment/dist/runtime_manifest.json
cp deployment/server.env.example deployment/server.env
# 用服务器编辑器填入密钥；保持现有线上参数，切勿提交 server.env
chmod 600 deployment/server.env
docker compose -f deployment/compose.yaml up -d --build
docker compose -f deployment/compose.yaml ps
curl --fail http://127.0.0.1:8501/_stcore/health
```

上述 python3 校验脚本仅使用标准库，不安装全局 Python 包；应用使用容器自己的 `/app/.venv`。
容器端依赖沿用本项目版本，PyTorch 单独指定 CPU wheel；Linux 构建尚需在有 Docker 的服务器验证。
运行数据以只读卷挂载，更新代码不会重建历史索引。

## 访问与验收

默认仅绑定服务器回环地址，尚未开放公网。在本机 PowerShell 运行：

```powershell
ssh -N -L 8501:127.0.0.1:8501 用户名@服务器IP
```

然后访问 http://127.0.0.1:8501 。若本机 8501 已占用，可将 `-L` 的第一个端口换成 8502。
课堂共享链接应在确定服务器与域名后配置 HTTPS、WebSocket 反向代理及访问限制；当前没有公开网址。
组员操作服务器时不要共享管理员密码或把 API Key 放进对话。

先验收三个页面、两种检索、V1/V2 模式及连续提问；没有 API Key 时先验证检索与 Dashboard。
真实问答使用服务器账号余额，只有手动提交问题才会调用 DeepSeek。
查看运行日志：`docker compose -f deployment/compose.yaml logs --tail=100`。
停止：`docker compose -f deployment/compose.yaml down`。
更新：`git pull` 后再次执行 `up -d --build`。
