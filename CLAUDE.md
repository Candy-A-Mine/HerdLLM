# 用户核心偏好 (全局生效)

## 身份与环境
- **操作系统**: Arch Linux (x86_64, Kernel: Linux)
- **Shell**: Zsh
- **语言**: 必须使用**简体中文**回答。

## Python 开发规则
- **包管理**: 必须使用 `poetry`。禁止使用 `pip install`。
- **环境**: 虚拟环境位于 `.venv/` 或由 Poetry 管理。
- **运行**: 使用 `poetry run python ...`。

## LaTeX 论文规则
- **编译**: 使用 `latexmk`。不要手动跑 pdflatex。
- **清理**: 编译产物在 `build/` 目录，PDF 在根目录。

## 行为准则
- **简洁**: 不要说废话，直接给代码或执行命令。
- **检查**: 在修改代码前，先确认当前 Git 分支状态。
