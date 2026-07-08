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
}

export interface SkillResult {
  skill: string;
  reply: string;
  data?: unknown;
}

export interface Skill {
  name: string;
  description: string;
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
