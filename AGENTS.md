# Agent instructions

These guidelines apply to every agent (Codex, Claude Code, and any other),
including subagents and other parallel agents. Claude Code loads this file
through `.claude/CLAUDE.md`.

# Getting session context
## STATE.md
 - Read .context/STATE.md to get the context. Do not read the project files unless explicitly asked to do so.
 - If you have to read the project files to get something done. Ask for permission.
 - After making changes, update STATE.md so that next session can pickup from there if needed.
 - Keep STATE.md as concise as possible. You can do this by removing past state information actively from it and only keeping info that is relevant currently in it.

## Planning
- When planning a new feature or making a change, always try to re-use existing code. 
- When you have to write new code always try to ensure that it is modular and generic so fits well with existing framework.

## HANDOFF.md
 - Use .context/HANDOFF.md to save some context for unfinished tasks. Keep this file as concise as possible.
 - Always include the date of each update in it.

## Making Local changes
- Whenever any change in the codebase is requested, even if a small change, do not implement it immediately unless explicitly asked to do so. First create a plan as ask to review it before implementing it.
- Unless explicitly told not to do so, you can make changes in the current checkout branch. You don't have to create some background worktree for that.
- Never commit changes by yourself. Always ask permission to do so.
- Individual test cases should not be modified unless absolutely necessary. When you have to modify an existing test, always ask for permission.

### Writing new tests
- Make sure that the test cases use only the public api wherever possible. Do not use internal members if there is an alternative.
- Do not add very implementation specific test cases. Wherever possible add future agnostic tests that test the intended functionality rather than whether the component behaves in an implementation specific way.
- Every builtin component must have some serialization and deserialization tests.

## Running tests
- After a major change, run the focused tests, then the full parallel suite:
  `python -m pytest -n 8 --dist loadfile`.
- Before every commit the full parallel suite must pass; run the serial
  suite only before merging to dev.
- Report failures with their output. Diagnose before changing anything;
  never weaken, skip or delete a test to make the suite pass.

## Running commands
- Unless required, run all commands in silent mode without any verbose, wherever possible. If the detailed output log of a command is not needed, suppress it.
- If a command requires GPU, then prioritize using one which has no processes.

## Learning
- When you learn something generic from your sessions, add it to .context/LEARNING.md.
