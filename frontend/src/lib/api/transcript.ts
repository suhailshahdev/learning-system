/**
 * Transcript API client.
 *
 * Mirrors backend/app/schemas/transcript_api.py. One query hook
 * for GET /api/sessions/{id}/transcript. Transcript data is
 * immutable for a given (terminal-state) session: once a session
 * is COMPLETED or ABANDONED, its turns never change. Caching is
 * therefore safe and the default staleTime is fine.
 */
import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { z } from "zod";

import { apiFetch } from "@/lib/api/client";
import {
    ParsedGradingSchema,
    ParsedSessionEndSchema,
    ParsedTurnSchema,
    SessionResponseSchema,
} from "@/lib/api/sessions";

// Entry-shape schemas mirror app/schemas/transcript_api.py.
// Discriminated on `kind` so consumers pattern-match exhaustively
// in switch/case blocks.

export const TurnEntrySchema = z.object({
    kind: z.literal("turn"),
    turn_index: z.number().int().nonnegative(),
    turn: ParsedTurnSchema,
});
export type TurnEntry = z.infer<typeof TurnEntrySchema>;

export const UserAnswerEntrySchema = z.object({
    kind: z.literal("user_answer"),
    turn_index: z.number().int().nonnegative(),
    answer: z.string(),
});
export type UserAnswerEntry = z.infer<typeof UserAnswerEntrySchema>;

export const GradingEntrySchema = z.object({
    kind: z.literal("grading"),
    turn_index: z.number().int().nonnegative(),
    grading: ParsedGradingSchema,
});
export type GradingEntry = z.infer<typeof GradingEntrySchema>;

export const SessionEndEntrySchema = z.object({
    kind: z.literal("session_end"),
    turn_index: z.number().int().nonnegative(),
    session_end: ParsedSessionEndSchema,
});
export type SessionEndEntry = z.infer<typeof SessionEndEntrySchema>;

export const TranscriptEntrySchema = z.discriminatedUnion("kind", [
    TurnEntrySchema,
    UserAnswerEntrySchema,
    GradingEntrySchema,
    SessionEndEntrySchema,
]);
export type TranscriptEntry = z.infer<typeof TranscriptEntrySchema>;

export const TranscriptResponseSchema = z.object({
    session: SessionResponseSchema,
    entries: z.array(TranscriptEntrySchema),
});
export type TranscriptResponse = z.infer<typeof TranscriptResponseSchema>;

export const transcriptKeys = {
    all: () => ["transcript"] as const,
    detail: (id: string) => ["transcript", id] as const,
};

async function fetchTranscript(
    sessionId: string,
    signal: AbortSignal,
): Promise<TranscriptResponse> {
    return apiFetch(`/sessions/${sessionId}/transcript`, {
        schema: TranscriptResponseSchema,
        signal,
    });
}

export function useTranscript(
    sessionId: string | undefined,
): UseQueryResult<TranscriptResponse, Error> {
    return useQuery({
        queryKey: transcriptKeys.detail(sessionId ?? ""),
        queryFn: ({ signal }) => {
            if (sessionId === undefined) {
                throw new Error("session id is required");
            }
            return fetchTranscript(sessionId, signal);
        },
        enabled: sessionId !== undefined,
    });
}
