/**
 * Per-user conversation state + a pluggable store.
 *
 * The handbook uses a bare in-process Map (lost on restart, not shareable). We
 * keep the in-memory default but hide it behind a SessionStore interface with TTL,
 * so a SQLite/MySQL-backed store can drop in later without touching the agent (乙).
 *
 * We deliberately store only the current search filter + last results — not a
 * sprawling user profile (red-zone decision: just one saved search per user).
 */
import type { SearchFilter } from '../search/filters.js';
import type { ListingRow } from '../search/listingRow.js';

export interface UserSession {
  filter: SearchFilter;
  lastResults?: ListingRow[];
  step: number;
  updatedAt: number;
}

export interface SessionStore {
  get(userId: string): Promise<UserSession | null>;
  set(userId: string, session: UserSession): Promise<void>;
  clear(userId: string): Promise<void>;
}

export function freshSession(): UserSession {
  return { filter: {}, step: 0, updatedAt: Date.now() };
}

/** Default in-memory store with idle TTL (entries expire after `ttlMs`). */
export class InMemorySessionStore implements SessionStore {
  private readonly map = new Map<string, UserSession>();
  constructor(private readonly ttlMs = 30 * 60_000) {}

  async get(userId: string): Promise<UserSession | null> {
    const s = this.map.get(userId);
    if (!s) return null;
    if (Date.now() - s.updatedAt > this.ttlMs) {
      this.map.delete(userId); // expired
      return null;
    }
    return s;
  }

  async set(userId: string, session: UserSession): Promise<void> {
    this.map.set(userId, { ...session, updatedAt: Date.now() });
  }

  async clear(userId: string): Promise<void> {
    this.map.delete(userId);
  }
}

/** Process-wide default; swap via dependency injection in tests / future stores. */
export const defaultSessionStore: SessionStore = new InMemorySessionStore();
