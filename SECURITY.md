# Security policy

## Reporting a vulnerability

Please do not disclose suspected vulnerabilities in a public issue, discussion,
benchmark artifact, or pull request.

Use the repository host's private vulnerability-reporting feature when it is
available. If private reporting is not enabled, contact a current repository
maintainer through the contact method shown in the repository metadata and
request a private channel before sharing technical details.

A useful report includes:

- affected revision or commit;
- a minimal, non-destructive reproduction;
- impact and affected configuration;
- proposed mitigation, if known.

Never include API keys, cookies, bearer tokens, private task data, production
logs, or account material in a report.

## Scope

The maintained code is the current development branch. This repository is
research software and its benchmark integrations can invoke local Docker,
Harbor, and provider tooling. Review local configuration carefully before
running non-dry commands.

## Handling

Maintainers will acknowledge a private report, assess reproducibility and
impact, and coordinate a fix or disclosure plan. Public acknowledgement should
not reveal a vulnerability before users have a reasonable opportunity to
update.
