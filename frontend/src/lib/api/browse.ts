/**
 * Sessions browse API client.
 *
 * Mirrors backend/app/schemas/browse_api.py. One query hook for
 * GET /api/sessions with optional state filter. Used by the
 * browse page.
 *
 * Cache key includes the state filter so switching tabs hits a
 * fresh fetch the first time and cache thereafter. Default
 * staleTime is fine since list contents change only on session
 * state transitions (start, approve, abandon).
 */
import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { z } from "zod";

import { apiFetch } from "@/lib/api/client";
import {
    LearningModeSchema,
    SessionStateSchema,
    TransportKindSchema,
} from "@/lib/api/sessions";

export const BrowseSessionRowSchema = z.object({
    id: z.string(),
    topic_id: z.string().nullable(),
    topic_path: z.string().nullable(),
    state: SessionStateSchema,
    transport_kind: TransportKindSchema,
    mode_used: LearningModeSchema,
    learned_item_count: z.number().int().nonnegative(),
    created_at: z.string(),
    updated_at: z.string(),
});
export type BrowseSessionRow = z.infer<typeof BrowseSessionRowSchema>;

export const BrowseResponseSchema = z.object({
    rows: z.array(BrowseSessionRowSchema),
    limit_reached: z.boolean(),
});
export type BrowseResponse = z.infer<typeof BrowseResponseSchema>;

// State filter as a TypeScript-side literal type. "all" is the
// frontend representation of "no state filter sent to backend."
// SessionState (the backend enum) covers the other four values.
export type BrowseStateFilter = "all" | z.infer<typeof SessionStateSchema>;

export const browseKeys = {
    all: () => ["browse"] as const,
    list: (state: BrowseStateFilter) => ["browse", state] as const,
};

async function fetchBrowse(
    state: BrowseStateFilter,
    signal: AbortSignal,
): Promise<BrowseResponse> {
    const search = state === "all" ? "" : `?state=${state}`;
    return apiFetch(`/sessions${search}`, {
        schema: BrowseResponseSchema,
        signal,
    });
}

export function useBrowse(
    state: BrowseStateFilter,
): UseQueryResult<BrowseResponse, Error> {
    return useQuery({
        queryKey: browseKeys.list(state),
        queryFn: ({ signal }) => fetchBrowse(state, signal),
    });
}
