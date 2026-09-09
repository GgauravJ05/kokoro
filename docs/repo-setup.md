# Repository setup and governance

What is configured, what is deferred, and how to finish the job.

## Applied now

| Setting | Value | Why |
|---|---|---|
| Visibility | **private** | Polishing before the work is public |
| Merge commits | disabled | Keeps `main` linear and bisectable |
| Squash / rebase merge | enabled | One commit per PR |
| Delete branch on merge | enabled | No stale branch accumulation |
| Auto-merge | enabled | Merge on green without babysitting |
| Commit signoff (web) | required | DCO-style attribution trail |
| Dependabot alerts | enabled | |
| Dependabot security fixes | enabled | Automated patch PRs |
| Dependabot version updates | `.github/dependabot.yml` | Weekly pip, monthly actions |
| Issues | enabled, templated | Bug / feature / research question |
| Wiki, Projects | disabled | Docs live in the repo |

## Deferred: branch protection

GitHub gates **repository rulesets behind a public repo or a paid plan**, so the
policy could not be applied to a free private repository. It is not lost — it is
checked in at [`.github/rulesets/main-protection.json`](../.github/rulesets/main-protection.json)
and applied by one command the moment the repo goes public:

```bash
gh repo edit GgauravJ05/kokoro --visibility public --accept-visibility-change-consequences
./scripts/setup_branch_protection.sh
```

That ruleset makes `main` require:

- a pull request with **1 approving review**
- **CODEOWNERS** review — every path is owned, so nothing merges unreviewed
- **all CI checks green** and the branch up to date with `main`
- **linear history** and **signed commits**
- **no force pushes**, **no deletion**

Until then, `main` is technically pushable directly. The pre-commit hook
`no-commit-to-branch` blocks accidental local commits to `main`, which covers the
realistic failure mode for a solo repo.

## Deferred: CodeQL

The CodeQL workflow is committed and correct, but Advanced Security is
unavailable on a free private repository. The job is guarded by

```yaml
if: github.event_name == 'schedule' || github.event.repository.visibility == 'public'
```

so it skips cleanly now and starts running the moment visibility flips — no red
badge in the meantime.

## Going public: the checklist

1. `make check` green
2. `kokoro benchmark` on the real corpus, results table pasted into the README
3. `gh repo edit --visibility public --accept-visibility-change-consequences`
4. `./scripts/setup_branch_protection.sh`
5. Confirm the CI, CodeQL and licence badges render
6. Tag `v0.1.0` — the release workflow builds, validates and attaches a
   provenance record

## Attribution surfaces

Attribution is deliberately redundant, so no single deletion removes it:

| Surface | File |
|---|---|
| Licence | [`LICENSE`](../LICENSE) — AGPL-3.0 |
| Attribution terms | [`NOTICE`](../NOTICE) |
| Academic citation | [`CITATION.cff`](../CITATION.cff) |
| Per-file headers | every `.py`, CI-enforced by `scripts/check_headers.py` |
| Runtime record | `kokoro provenance` |
| Artifact stamps | `kokoro.provenance.stamp()` on every report and checkpoint |
| Embedding watermark | `kokoro watermark <file.npy>` |
| Service headers | `X-Kokoro-Provenance` on every API response |
| Ownership | [`.github/CODEOWNERS`](../.github/CODEOWNERS) |
