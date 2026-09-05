/**
 * Skill abstraction (Week 9).
 *
 * A "skill" is a self-contained capability module behind a uniform interface. A
 * DETERMINISTIC router (not an autonomous LLM loop) picks which skill(s) to run —
 * honest naming: these are skills/tools dispatched by a router, not "agents".
 * The `description` + input shape are carried so a future LLM router (reserved
 * slot) could pick by description, or so we could emit OpenClaw SKILL.md wrappers.
 */
import type { SearchFilter } from '../search/filters.js';
import type { LLMClient } from '../llm/client.js';

export interface SkillContext {
  userId: string;
  message: string;
  /** Params extracted by the router (reused from parseQuery). */
  filter: SearchFilter;
  /** The configured LLM (so skills' own parsing uses the same fallback). */
  llm?: LLMClient;
  /** In a multi-skill plan, the outputs of skills already run (the parallel batch +
   * earlier serial steps). Available to serially-run skills so a dependent one can use them. */
  priorResults?: SkillResult[];
  /** Auto/agent mode only: the raw structured tool arguments the LLM supplied (beyond
   * what maps to `filter`) — e.g. a custom email subject/body. Undefined in router mode,
   * so deterministic behavior is unchanged. */
  args?: Record<string, unknown>;
}

export interface SkillResult {
  skill: string;
  reply: string;
  data?: unknown;
}

export interface Skill {
  name: string;
  description: string;
  /** OPT-IN parallelism: set true ONLY when this skill is verified independent +
   * concurrency-safe (no shared mutable state, no dependency on other skills' output).
   * Such skills run together in a PARALLEL batch. Anything left unmarked runs
   * SEQUENTIALLY (the safe default) after that batch, and can read ctx.priorResults —
   * so "unsure whether it's safe to parallelize" => don't mark it => serial. */
  parallelSafe?: boolean;
  /** Auto/agent mode: JSON Schema for THIS tool's parameters. Falls back to a shared
   * schema when absent — lets a skill (e.g. email) declare bespoke params like subject/body. */
  paramSchema?: Record<string, unknown>;
  run(ctx: SkillContext): Promise<SkillResult>;
}

export class SkillRegistry {
  private readonly skills = new Map<string, Skill>();

  register(skill: Skill): this {
    this.skills.set(skill.name, skill);
    return this;
  }

  get(name: string): Skill | undefined {
    return this.skills.get(name);
  }

  has(name: string): boolean {
    return this.skills.has(name);
  }

  list(): Skill[] {
    return [...this.skills.values()];
  }
}
