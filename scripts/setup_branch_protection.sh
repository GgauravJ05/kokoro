#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gaurav Jadhav <gauravmakarandjadhav@gmail.com>
#
# Apply the `main` branch ruleset.
#
# GitHub gates repository rulesets behind a public repository or a paid plan, so
# this cannot run while the repo is private on a free account. Run it the moment
# the repo goes public:
#
#     gh repo edit GgauravJ05/kokoro --visibility public --accept-visibility-change-consequences
#     ./scripts/setup_branch_protection.sh
#
# The ruleset itself lives in .github/rulesets/main-protection.json so the
# protection policy is reviewable in version control rather than buried in the
# web UI.

set -euo pipefail

REPO="${1:-GgauravJ05/kokoro}"
RULESET="$(dirname "$0")/../.github/rulesets/main-protection.json"

echo "Applying branch protection to ${REPO}…"

visibility=$(gh api "repos/${REPO}" --jq '.visibility')
if [[ "$visibility" == "private" ]]; then
  echo
  echo "  WARNING: ${REPO} is private."
  echo "  Rulesets need a public repo or GitHub Pro. This will very likely 403."
  echo
fi

if existing=$(gh api "repos/${REPO}/rulesets" --jq '.[] | select(.name=="main-protection") | .id' 2>/dev/null) \
   && [[ -n "$existing" ]]; then
  echo "Updating existing ruleset ${existing}…"
  gh api -X PUT "repos/${REPO}/rulesets/${existing}" --input "$RULESET" --jq '.name + " updated"'
else
  gh api -X POST "repos/${REPO}/rulesets" --input "$RULESET" --jq '.name + " created"'
fi

echo
echo "main now requires:"
echo "  • a pull request with 1 approving review"
echo "  • CODEOWNERS review (you, on every path)"
echo "  • all CI checks green and up to date with main"
echo "  • linear history, signed commits"
echo "  • no force pushes, no deletion"
