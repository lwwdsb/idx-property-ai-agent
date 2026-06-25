/**
 * Minimal tool (handbook Week 1 example), typed.
 * A "tool" in this codebase is a typed async function the agent can call.
 */
export interface CurrentTime {
  currentTime: string;
}

export async function getCurrentTime(): Promise<CurrentTime> {
  return { currentTime: new Date().toISOString() };
}
