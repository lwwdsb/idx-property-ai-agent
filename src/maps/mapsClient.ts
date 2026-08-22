/**
 * Maps client — commute/proximity via Google Maps (Geocoding + Distance Matrix).
 *
 * A thin, injectable client used ONLY inside the search skill, when the parse layer
 * produced a `proximity` slot. The DECISION to call it is made by CODE (slot present?),
 * never by the LLM choosing a tool — the LLM/regex only EXTRACTED the slot.
 *
 * 乙: any failure (no key, API error) degrades gracefully — the caller keeps the
 * un-ranked search results instead of erroring.
 */
import { config } from '../config.js';
import { logger } from '../logger.js';

export interface LatLng { lat: number; lng: number; }
export interface CommuteResult { id: number; minutes: number | null; }

export interface MapsClient {
  readonly available: boolean;
  /** Resolve a place/address text to coordinates (null if not found). */
  geocode(place: string): Promise<LatLng | null>;
  /** Commute minutes from each origin (with its listing id) to the destination. */
  commuteMinutes(
    origins: Array<{ id: number; lat: number; lng: number }>,
    dest: LatLng,
    mode: 'driving' | 'transit' | 'walking',
  ): Promise<CommuteResult[]>;
}

const GEOCODE = 'https://maps.googleapis.com/maps/api/geocode/json';
const MATRIX = 'https://maps.googleapis.com/maps/api/distancematrix/json';
const BATCH = 25;   // Distance Matrix origins-per-request cap

export function getMapsClient(): MapsClient {
  const key = config.maps.apiKey;
  if (!key) {
    return {
      available: false,
      async geocode() { return null; },
      async commuteMinutes() { return []; },
    };
  }
  return {
    available: true,
    async geocode(place) {
      try {
        const url = `${GEOCODE}?address=${encodeURIComponent(place)}&key=${key}`;
        const r = await fetch(url, { signal: AbortSignal.timeout(8000) });
        const d = (await r.json()) as { status: string; results?: Array<{ geometry: { location: LatLng } }> };
        if (d.status !== 'OK' || !d.results?.length) return null;
        return d.results[0]!.geometry.location;
      } catch (e) {
        logger.warn('geocode failed', { place, error: String(e) });
        return null;
      }
    },
    async commuteMinutes(origins, dest, mode) {
      const out: CommuteResult[] = [];
      for (let i = 0; i < origins.length; i += BATCH) {
        const chunk = origins.slice(i, i + BATCH);
        try {
          const originsParam = chunk.map((o) => `${o.lat},${o.lng}`).join('|');
          const url = `${MATRIX}?origins=${encodeURIComponent(originsParam)}`
            + `&destinations=${dest.lat},${dest.lng}&mode=${mode}&key=${key}`;
          const r = await fetch(url, { signal: AbortSignal.timeout(10000) });
          const d = (await r.json()) as { rows?: Array<{ elements: Array<{ status: string; duration?: { value: number } }> }> };
          chunk.forEach((o, j) => {
            const el = d.rows?.[j]?.elements?.[0];
            out.push({ id: o.id, minutes: el?.status === 'OK' && el.duration ? Math.round(el.duration.value / 60) : null });
          });
        } catch (e) {
          logger.warn('distance matrix failed', { error: String(e) });
          chunk.forEach((o) => out.push({ id: o.id, minutes: null }));
        }
      }
      return out;
    },
  };
}
