# Security Policy

## Reporting a vulnerability

If you believe you have found a security vulnerability, please do not open a public issue.

Use [GitHub Security Advisories](https://github.com/dalozedidier-dot/AstroOracle/security/advisories/new)
or email the maintainer at dalozedidier@gmail.com.

Please include:

- a description of the issue
- steps to reproduce
- impact (data exposure, arbitrary code execution, etc.)

## Scope notes

The default `astrooracle serve` UI is a local annotation tool. Do not expose it
on the public internet without authentication. The retrain hook executes a
configured script; treat that path as trusted configuration.
