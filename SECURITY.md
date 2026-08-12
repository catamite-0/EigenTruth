# Security Policy

## Reporting A Vulnerability

Please report suspected security vulnerabilities privately before opening a public issue. Use GitHub's private vulnerability reporting feature for this repository when available. If private reporting is not available, contact the maintainer through the repository owner's GitHub profile and request a private disclosure channel.

Include:

- a description of the issue and its potential impact
- steps to reproduce or a proof of concept
- affected files, versions, or dependency versions
- any suggested mitigation

Please allow maintainers time to investigate and coordinate disclosure before publishing details.

## Scope

Security reports may include:

- vulnerabilities in EigenTruth source code
- unsafe handling of files, configuration values, or inputs used during model loading
- dependency vulnerabilities with a concrete impact on this project
- documentation that encourages unsafe execution of untrusted code or artifacts

For ordinary bugs, feature requests, or documentation improvements, use [GitHub Issues](https://github.com/catamite-0/EigenTruth/issues).

## Research-Preview Disclaimer

EigenTruth is an alpha-stage research toolkit. It is not a production safety system, a factuality guarantee, or a security control for deployed language models. Its diagnostics and activation-steering hooks must not be used as the sole protection for production systems.

## 安全说明

如发现潜在安全漏洞，请优先通过 GitHub 私有漏洞报告功能进行负责任披露，不要直接创建公开 issue。如果私有报告不可用，请通过仓库所有者的 GitHub 个人资料联系维护者，并请求建立私有披露渠道。

EigenTruth 是一个处于 alpha 阶段的研究工具库。它不是生产级安全系统，不提供事实性保证，也不能作为已部署语言模型的安全控制措施。诊断指标和激活引导 hook 不应成为生产系统的唯一保护手段。
