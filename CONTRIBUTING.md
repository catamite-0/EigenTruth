# Contributing To EigenTruth

Thank you for your interest in EigenTruth. Contributions are welcome, especially improvements that make experiments more reproducible, diagnostics easier to interpret, and project limitations clearer.

EigenTruth is an alpha-stage research preview. It is not a production safety system and does not prove that language-model outputs are true. Contributions, issues, and pull requests should preserve that distinction.

## Development Setup

EigenTruth requires Python 3.10 or newer.

```bash
git clone https://github.com/catamitez0-maker/EigenTruth.git
cd EigenTruth
python -m venv .venv
```

Activate the environment:

```bash
# POSIX shells
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.\.venv\Scripts\activate.bat
```

Install the package and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The core package intentionally keeps Hugging Face and dataset tooling optional. Add extras only when you need them:

```bash
python -m pip install -e ".[hf]"        # Hugging Face model-loading workflows
python -m pip install -e ".[examples]"  # runnable examples
python -m pip install -e ".[eval]"      # benchmark datasets
```

If your environment needs a specific PyTorch build, install it first using the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

## Tests And Lint

Run the full unit suite:

```bash
python -m pytest tests/ -v
```

Run lint checks:

```bash
python -m ruff check src tests examples benchmarks
```

Or run the common checks with the local Makefile, which auto-detects `.venv/bin/python` when present:

```bash
make check
make release-check  # also builds the package
```

Before opening a pull request, run both commands from the repository root. Add tests for behavior changes and update documentation when interfaces, experiment assumptions, or limitations change.

## Pull Request Process

1. Fork the repository or create a focused branch.
2. Keep the change small enough to review clearly.
3. Add or update tests when behavior changes.
4. Run the test and lint commands above.
5. Describe the motivation, implementation, validation, and research caveats in the pull request.
6. Link related issues when applicable.

Suggested commit format:

```text
<type>: <description>

[optional body]
```

Common types include `feat`, `fix`, `docs`, `test`, `refactor`, and `ci`.

## Code Guidelines

- Use `ruff` for style and import checks.
- Add concise docstrings to public functions and classes.
- Keep numerical operations explicit about dtype, device, and batch behavior.
- Preserve backwards compatibility unless a breaking change is discussed first.
- Treat example outputs as qualitative observations unless a reproducible evaluation supports a stronger claim.

## Reporting Issues

Use [GitHub Issues](https://github.com/catamitez0-maker/EigenTruth/issues) for reproducible bugs, documentation gaps, feature proposals, and improvements to experiment quality.

Include:

- operating system and Python version
- installed PyTorch and Transformers versions
- model identifier, target layer, and relevant configuration
- a minimal reproduction when possible
- expected and observed behavior
- logs or stack traces with sensitive information removed

Do not use public issues for undisclosed security vulnerabilities. Follow [`SECURITY.md`](SECURITY.md) instead.

## Research Disclaimer

EigenTruth exposes experimental representation diagnostics and activation-steering hooks. A lower distance, lower HSE value, or changed output is not proof of factuality, safety, or robustness. Research contributions should state their assumptions and avoid production-safety claims.

---

# EigenTruth 贡献指南

感谢你对 EigenTruth 的关注。我们欢迎各种贡献，尤其是能够提高实验可复现性、增强诊断可解释性和明确项目局限性的改进。

EigenTruth 是一个处于 alpha 阶段的研究预览项目。它不是生产级安全系统，也不能证明语言模型输出为真。贡献、issue 和 pull request 都应保持这一区分。

## 开发环境

EigenTruth 需要 Python 3.10 或更高版本。

```bash
git clone https://github.com/catamitez0-maker/EigenTruth.git
cd EigenTruth
python -m venv .venv
```

激活虚拟环境：

```bash
# POSIX shell
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows 命令提示符
.\.venv\Scripts\activate.bat
```

安装项目和开发依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

核心包会尽量保持轻量；Hugging Face 与数据集工具按需安装：

```bash
python -m pip install -e ".[hf]"        # Hugging Face 模型加载流程
python -m pip install -e ".[examples]"  # 可运行示例
python -m pip install -e ".[eval]"      # benchmark 数据集
```

如果环境需要特定版本的 PyTorch，请先按照 [pytorch.org](https://pytorch.org/get-started/locally/) 的说明安装。

## 测试与代码检查

```bash
python -m pytest tests/ -v
python -m ruff check src tests examples benchmarks
```

也可以使用本地 Makefile 运行常用检查；存在 `.venv/bin/python` 时会自动优先使用它：

```bash
make check
make release-check  # 也会构建包
```

提交 pull request 前，请在仓库根目录运行以上命令。行为发生变化时应补充测试；接口、实验假设或局限性发生变化时应更新文档。

## Pull Request 流程

1. Fork 仓库或创建范围明确的分支。
2. 保持改动规模适中，便于清晰审查。
3. 行为变化时添加或更新测试。
4. 运行测试和 lint 命令。
5. 在 pull request 中说明动机、实现、验证步骤和研究局限。
6. 如有相关 issue，请添加链接。

建议的 commit 格式：

```text
<type>: <description>

[optional body]
```

常用类型包括 `feat`、`fix`、`docs`、`test`、`refactor` 和 `ci`。

## Issue 报告

可使用 [GitHub Issues](https://github.com/catamitez0-maker/EigenTruth/issues) 报告可复现 bug、文档缺口、功能建议和实验质量改进。

请包含操作系统、Python 版本、PyTorch 和 Transformers 版本、模型标识、目标层、相关配置、最小复现方式、预期行为和实际行为。请移除日志或堆栈信息中的敏感内容。

不要通过公开 issue 报告尚未披露的安全漏洞。请遵循 [`SECURITY.md`](SECURITY.md)。

## 研究免责声明

EigenTruth 提供实验性表征诊断和激活引导 hook。更低的距离、更低的 HSE 值或发生变化的输出都不能证明事实性、安全性或鲁棒性。研究贡献应明确说明假设，并避免生产安全声明。
