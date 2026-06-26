# Security Policy

## Scope

This project is a safety layer — vulnerabilities here could allow harmful content to reach users. Issues in scope include:

- Bypasses of Stage 1 (pre-check) or Stage 2 (safety checker) that allow clearly harmful content through
- Prompt injection patterns that cause the pipeline to misclassify its own safety verdict
- Information leakage from the error response that reveals which stage blocked a request or why

Out of scope: model-level jailbreaks of the underlying LLM (report those to the model provider).

## Reporting a vulnerability

**Do not open a public GitHub Issue for security vulnerabilities.**

Use [GitHub's private security advisory](../../security/advisories/new) to report the issue confidentially. Include:
- A description of the vulnerability
- Steps to reproduce (a minimal example prompt or config)
- The version/commit you tested against

I aim to respond within 72 hours and will coordinate a fix before any public disclosure.
