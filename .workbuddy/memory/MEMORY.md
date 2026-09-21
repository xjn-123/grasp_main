# robot_grasp_test 项目长期笔记

## 开发环境约定

- 包管理与环境：uv，`uv sync` 后项目以 editable 方式装入 `.venv`（`grasp.pth` → `src`），Python 3.11.14。
- VS Code 必须从 `robot_grasp_test` 这一层打开；父目录 `arm_disorderly_swap` 没有 venv / pyproject.toml。
- `.vscode/settings.json` 里显式写了 `python.defaultInterpreterPath` 和 `python.analysis.extraPaths: ["src"]`。

## 已知坑

- **用户口中的"补全"指的是 AI 整行/多行幽灵文本续写**（inline suggestion），不是 VS Code 的建议列表。Pylance 只负责后者；前者要装 AI 插件（Copilot / 通义灵码 / Supermaven 等）。沟通时注意区分这两个概念。

- **Pylance 解析不了 editable install 的 `grasp.pth`**：日志报 `No sandbox available ... Dynamic pth file resolution requires Python 3.13 or later`。靠 `extraPaths: ["src"]` 兜住，别删这条配置。
- **机器上有个全局 Python 3.8.0**（`AppData\Local\Programs\Python\Python38`），Pylance 会用它起 `<default>` 服务。工作区外的文件（或没开文件夹的窗口）都会被 3.8 接管，导致第三方库补全缺失。
- **Pylance 2026.3.1 的日志通道不叫 "Pylance"**，叫 `Python Language Server`（中文界面显示「Python 语言服务器」）。排查时看这个通道。
- `requirements.txt` 是 UTF-16 编码的二进制文件，建议转成 UTF-8。

## 相关项目

- `../robot_grasp_show`：老项目，无 `.venv`、无 `.vscode`，被 Python 3.8.0 接管，技术债。
