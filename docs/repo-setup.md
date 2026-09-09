# Repository setup and governance

> The research plan this repo implements is in [`plan.html`](plan.html) —
> a standalone page, no build step, just open it in a browser.

What is configured, what is deferred, and how to finish the job.

## Applied now

| Setting | Value | Why |
|---|---|---|
| Visibility | **public** | Applied 2026-09-09 |
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

## Branch protection — applied, and corrected

The ruleset in [`.github/rulesets/main-protection.json`](../.github/rulesets/main-protection.json)
is active on `main`. It requires:

- a **pull request** for every change
- **all CI checks green** and the branch up to date with `main`
- **linear history**
- **no force pushes**, **no deletion**

### Two rules were removed, and why

The ruleset as originally written was unworkable for a sole maintainer, and
applying it briefly locked the repository:

**`required_signatures`.** Every commit here is unsigned and commit signing is
not configured, so this rule rejects *every* push to `main`. Enabling it means
first setting up SSH or GPG signing and registering the public key with GitHub
as a signing key — worth doing, but it is a prerequisite, not a side effect.

**`required_approving_review_count: 1`** plus **`require_code_owner_review`.**
GitHub does not permit approving your own pull request. With one maintainer and
no bypass actors, nothing could ever be merged. The pull-request requirement is
kept at **zero** approvals, which still routes every change through a PR and its
status checks — the part that actually protects the branch.

Restore either rule the moment a second maintainer exists, or once signing is
configured.

### Enabling signed commits later

```bash
ssh-keygen -t ed25519 -C "signing" -f ~/.ssh/git_signing
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/git_signing.pub
git config --global commit.gpgsign true
# then add ~/.ssh/git_signing.pub to GitHub as a SIGNING key (not an auth key)
```

Afterwards, re-add the `required_signatures` rule to the ruleset JSON and rerun
`./scripts/setup_branch_protection.sh`.

## CodeQL

Now active. The workflow was guarded on repository visibility while the repo was
private (Advanced Security is unavailable on free private repos), and that guard
now passes, so CodeQL runs on push and weekly.

## Going public: done

1. ✅ `make check` green
2. ✅ Results measured and reported, including the negative ones
3. ✅ `gh repo edit --visibility public`
4. ✅ `./scripts/setup_branch_protection.sh` (then corrected — see above)
5. ⬜ Confirm the CI, CodeQL and licence badges render
6. ⬜ Tag `v0.1.0` — the release workflow builds, validates and attaches a
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
