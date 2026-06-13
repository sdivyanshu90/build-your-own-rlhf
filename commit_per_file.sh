#!/usr/bin/env bash
#
# commit_per_file.sh — stage, commit, and push every changed file individually,
# each with its own descriptive (Conventional Commits) message.
#
# USAGE
#   ./commit_per_file.sh                 # commit + push each file to origin/<current branch>
#   PUSH=false ./commit_per_file.sh      # commit each file but DO NOT push
#   DRY_RUN=true ./commit_per_file.sh    # print what would happen, change nothing
#   BRANCH=feat/rlhf ./commit_per_file.sh# create/switch to BRANCH first, then commit there
#   REMOTE=upstream ./commit_per_file.sh # push to a remote other than 'origin'
#
# NOTES
#   * Files ignored by .gitignore (outputs/, coverage.xml, site/, *.pt, ...) are
#     skipped automatically.
#   * Pushing happens after EACH commit (as requested). Set PUSH=false to push
#     yourself at the end with a single `git push`.
#   * Requires a configured remote and (for a brand-new branch) push permission.
#
set -euo pipefail

REMOTE="${REMOTE:-origin}"
PUSH="${PUSH:-true}"
DRY_RUN="${DRY_RUN:-false}"
BRANCH="${BRANCH:-}"

cd "$(git rev-parse --show-toplevel)"

# Optionally create/switch to a dedicated branch before committing.
if [[ -n "$BRANCH" ]]; then
  git checkout -B "$BRANCH"
fi
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# --- Map a path to a descriptive commit message --------------------------------
desc() {
  local f="$1"
  local base
  base="$(basename "$f")"
  case "$f" in
    pyproject.toml)            echo "build: project metadata, tooling, and coverage gate" ;;
    Makefile)                  echo "build: developer task Makefile" ;;
    README.md)                 echo "docs: project README with architecture and runbook" ;;
    SECURITY.md)               echo "docs: security policy and vulnerability reporting" ;;
    .gitignore)                echo "chore: ignore build, test, and training artifacts" ;;
    commit_per_file.sh)        echo "chore: per-file commit helper script" ;;
    rlhf_ppo_prompt.md)        echo "docs: original RLHF-PPO specification prompt" ;;

    src/rlhf/exceptions.py)    echo "feat(core): typed exception hierarchy" ;;
    src/rlhf/utils.py)         echo "feat(core): seeding and reproducibility utilities" ;;
    src/rlhf/__main__.py)      echo "feat(cli): Typer command-line entrypoint" ;;
    src/rlhf/py.typed)         echo "chore(core): PEP 561 inline-typing marker" ;;
    src/rlhf/__init__.py)      echo "feat(core): package public surface" ;;

    src/rlhf/config/defaults/*) echo "feat(config): default hyperparameters for ${base%.yaml}" ;;
    src/rlhf/config/schema.py) echo "feat(config): Pydantic configuration schema with validators" ;;
    src/rlhf/config/*)         echo "feat(config): $base" ;;

    src/rlhf/data/schemas.py)  echo "feat(data): Pydantic data schemas (Preference/Prompt/Rollout)" ;;
    src/rlhf/data/*)           echo "feat(data): $base" ;;

    src/rlhf/models/value_head.py)      echo "feat(models): PPO value head (zero-init MLP)" ;;
    src/rlhf/models/policy.py)          echo "feat(models): policy model with value head and generation" ;;
    src/rlhf/models/reward_model.py)    echo "feat(models): reward model, Bradley-Terry loss, ensemble" ;;
    src/rlhf/models/reference_model.py) echo "feat(models): frozen reference model for the KL penalty" ;;
    src/rlhf/models/base.py)            echo "feat(models): shared model interface and tensor helpers" ;;
    src/rlhf/models/*)                  echo "feat(models): $base" ;;

    src/rlhf/training/ppo/algorithm.py)    echo "feat(ppo): clipped-surrogate PPO loss" ;;
    src/rlhf/training/ppo/rollout.py)      echo "feat(ppo): rollout buffer, reward shaping, vectorized GAE" ;;
    src/rlhf/training/ppo/kl_controller.py) echo "feat(ppo): adaptive and fixed KL controllers" ;;
    src/rlhf/training/ppo/trainer.py)      echo "feat(ppo): five-phase PPO training orchestrator" ;;
    src/rlhf/training/ppo/scheduler.py)    echo "feat(ppo): warmup learning-rate schedules" ;;
    src/rlhf/training/ppo/*)               echo "feat(ppo): $base" ;;
    src/rlhf/training/reward_model/*)      echo "feat(reward): $base" ;;
    src/rlhf/training/sft/*)               echo "feat(sft): $base" ;;
    src/rlhf/training/*)                   echo "feat(training): $base" ;;

    src/rlhf/inference/*)      echo "feat(inference): $base" ;;
    src/rlhf/evaluation/*)     echo "feat(eval): $base" ;;
    src/rlhf/monitoring/*)     echo "feat(monitoring): $base" ;;
    src/rlhf/security/*)       echo "feat(security): $base" ;;
    src/rlhf/distributed/*)    echo "feat(distributed): $base" ;;
    src/rlhf/*)                echo "feat: $base" ;;

    tests/fixtures/gpt2/*)     echo "test(fixtures): vendored GPT-2 tokenizer for offline tests" ;;
    tests/conftest.py)         echo "test: shared pytest fixtures and tiny-model factories" ;;
    tests/unit/*)              echo "test(unit): $base" ;;
    tests/integration/*)       echo "test(integration): $base" ;;
    tests/system/*)            echo "test(system): $base" ;;
    tests/adversarial/*)       echo "test(adversarial): $base" ;;
    tests/*)                   echo "test: $base" ;;

    scripts/*)                 echo "feat(scripts): $base entrypoint" ;;

    infra/Dockerfile)          echo "build(docker): multi-stage non-root training image" ;;
    infra/docker-compose.yml)  echo "infra: local dev compose stack" ;;
    infra/helm/*)              echo "infra(helm): $base" ;;
    infra/terraform/*)         echo "infra(terraform): $base" ;;
    infra/*)                   echo "infra: $base" ;;

    docs/adr/*)                echo "docs(adr): ${base%.md} architecture decision record" ;;
    docs/architecture/*)       echo "docs(arch): ${base%.md}" ;;
    docs/guides/*)             echo "docs(guide): ${base%.md}" ;;
    docs/mkdocs.yml)           echo "docs: MkDocs site configuration" ;;
    docs/*)                    echo "docs: $base" ;;

    .github/workflows/*)       echo "ci: ${base%.yml} workflow" ;;
    .github/CODEOWNERS)        echo "ci: code ownership rules" ;;
    .github/*)                 echo "ci: $base" ;;

    *)                         echo "chore: add $f" ;;
  esac
}

# --- Collect changed + untracked files (respecting .gitignore), sorted ---------
mapfile -d '' FILES < <(git ls-files -z --modified --others --exclude-standard | sort -z)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "Nothing to commit — working tree is clean."
  exit 0
fi

echo "About to commit ${#FILES[@]} file(s) individually on branch '$CURRENT_BRANCH'."
echo "PUSH=$PUSH  DRY_RUN=$DRY_RUN  REMOTE=$REMOTE"
echo

count=0
for f in "${FILES[@]}"; do
  [[ -z "$f" ]] && continue
  msg="$(desc "$f")"
  count=$((count + 1))
  printf '[%d/%d] %s\n        -> %s\n' "$count" "${#FILES[@]}" "$f" "$msg"
  if [[ "$DRY_RUN" == "true" ]]; then
    continue
  fi
  git add -- "$f"
  # Skip if staging produced no change (e.g. file unchanged since a prior run).
  if git diff --cached --quiet; then
    echo "        (no change staged, skipping)"
    continue
  fi
  git commit -q -m "$msg" \
    -m "Done committing $f."
  if [[ "$PUSH" == "true" ]]; then
    git push -q "$REMOTE" "$CURRENT_BRANCH"
  fi
done

echo
if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run complete — no commits were made."
elif [[ "$PUSH" == "true" ]]; then
  echo "Done: committed and pushed $count file(s) to $REMOTE/$CURRENT_BRANCH."
else
  echo "Done: committed $count file(s) locally. Push when ready:  git push $REMOTE $CURRENT_BRANCH"
fi
