# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| `main` | ✅ |
| tagged releases | latest only |

This is pre-1.0 research software. Only `main` receives fixes.

## Reporting a vulnerability

**Do not open a public issue.**

Use [GitHub private vulnerability reporting](https://github.com/GgauravJ05/kokoro/security/advisories/new),
or email **gauravmakarandjadhav@gmail.com** with `[SECURITY]` in the subject.

Please include the affected version or commit, reproduction steps, and what an
attacker gains. You will get an acknowledgement within 72 hours and an
assessment within 7 days. Please allow 90 days before public disclosure.

## Scope

In scope:

- Remote code execution via crafted API responses, model checkpoints, or
  configuration
- Deserialisation issues in checkpoint or index loading
- Credential or token leakage through logs, artifacts or provenance stamps
- Injection through the ingestion pipeline or the `/recommend` endpoint
- Denial of service in the serving path

Out of scope:

- Rate limiting on third-party APIs (a design constraint, not a vulnerability)
- Findings that require an already-compromised host
- Missing hardening headers on a local development server

## Notes for operators

- **Never load a checkpoint you did not produce.** `torch.load` executes pickled
  code. Verify the provenance stamp first: `kokoro provenance`.
- Keep `.env` out of version control — `.gitignore` already covers it.
- The service returns no user data and stores no credentials by design. If you
  add authentication, that is your threat model to own.
