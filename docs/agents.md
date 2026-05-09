# Agent skills

Repo-local skills live under [`../.claude/skills/`](../.claude/skills/) and
encode the workflow rules a future LLM agent should follow when working in
this tree.

The principle: skills route agents into the canonical docs and the goal list,
they do not duplicate them.

## Skills index

| Skill | Path | Triggers on | Routes to |
|---|---|---|---|
| `llm-kittens-port` | [`.claude/skills/llm-kittens-port/SKILL.md`](../.claude/skills/llm-kittens-port/SKILL.md) | "implement the next milestone", "add a TK kernel wrapper", "port X from llm.c", anything that mutates `llmc/` or `train_*.cu` | [`../goal.md`](../goal.md), [architecture.md](architecture.md), [kernel-reference.md](kernel-reference.md), [porting-notes.md](porting-notes.md) |

## When to use which doc

If you are an LLM agent landing in this repo:

1. **Always read [`../goal.md`](../goal.md) first.** It tells you what milestone
   you are in and what is done vs pending.
2. **For architectural questions**, read [architecture.md](architecture.md).
   The layering rule is non-negotiable — train source must not pull `kittens::`
   into scope.
3. **For "how do I implement <kernel>?"**, read [kernel-reference.md](kernel-reference.md)
   for the per-kernel mapping and [porting-notes.md](porting-notes.md) for the
   wrapper-PR checklist and common gotchas.
4. **For multi-GPU / multi-node**, read [multi-gpu.md](multi-gpu.md).
   ZeRO-0/1/2/3 and the NCCL init paths are wired; ZeRO-3 uses local
   parameter shards plus all-gather into the current full compute layout.
5. **For Llama-3**, read [llama3.md](llama3.md). GQA/RoPE remains the
   highest-risk component and has its own validation plan.
6. **For testing**, read [testing.md](testing.md). The tolerance table is the
   bit you'll re-derive most often.

## What skills should *not* do

- Skills should not contain code listings. Code lives in source files. The
  skill points at the file.
- Skills should not contain a milestone TODO. The TODO is in
  [`../goal.md`](../goal.md). The skill points at it.
- Skills should not be longer than a couple of pages. A long skill is a doc
  in disguise.
