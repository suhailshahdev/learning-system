/**
 * LLM-call observability API client.
 *
 * Mirrors backend/app/schemas/admin_api.py. Two query hooks:
 * useLLMCalls for GET /api/admin/llm-calls (filterable list) and
 * useLLMCallStats for GET /api/admin/llm-calls/stats (aggregates
 * over a window). Used by the observability page.
 *
 * Token and cost fields are nullable: claude.ai reports no tokens,
 * DeepSeek's usage is not yet threaded, and cost is unknown until
 * pricing lands. The page renders null as a dash.
 *
 * Datetimes stay ISO strings, not Date objects, matching the rest
 * of the API client.
 */
import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { z } from "zod";

import { apiFetch } from "@/lib/api/client";
import { TransportKindSchema } from "@/lib/api/sessions";

// "all" is the frontend representation of "no transport filter sent
// to backend." TransportKind covers the two real values.
export type TransportFilter = "all" | z.infer<typeof TransportKindSchema>;

export const LLMCallRowSchema = z.object({
    id: z.string(),
    trace_id: z.string(),
    session_id: z.string().nullable(),
    transport_kind: TransportKindSchema,
    method: z.string(),
    model: z.string().nullable(),
    latency_ms: z.number().int().nonnegative(),
    prompt_chars: z.number().int().nonnegative(),
    response_chars: z.number().int().nonnegative(),
    prompt_tokens: z.number().int().nonnegative().nullable(),
    completion_tokens: z.number().int().nonnegative().nullable(),
    cost_usd: z.number().nonnegative().nullable(),
    success: z.boolean(),
    error: z.string().nullable(),
    created_at: z.string(),
});
export type LLMCallRow = z.infer<typeof LLMCallRowSchema>;

export const LLMCallListResponseSchema = z.object({
    rows: z.array(LLMCallRowSchema),
    limit_reached: z.boolean(),
});
export type LLMCallListResponse = z.infer<typeof LLMCallListResponseSchema>;

export const LLMCallStatsSchema = z.object({
    window_days: z.number().int().positive(),
    total_calls: z.number().int().nonnegative(),
    error_count: z.number().int().nonnegative(),
    error_rate: z.number().min(0).max(1),
    latency_p50_ms: z.number().int().nonnegative().nullable(),
    latency_p95_ms: z.number().int().nonnegative().nullable(),
    total_cost_usd: z.number().nonnegative(),
});
export type LLMCallStats = z.infer<typeof LLMCallStatsSchema>;

export const llmCallKeys = {
    all: () => ["llm-calls"] as const,
    list: (transport: TransportFilter, success: boolean | null) =>
        ["llm-calls", "list", transport, success] as const,
    stats: (windowDays: number, transport: TransportFilter) =>
        ["llm-calls", "stats", windowDays, transport] as const,
};

async function fetchLLMCalls(
    transport: TransportFilter,
    success: boolean | null,
    signal: AbortSignal,
): Promise<LLMCallListResponse> {
    const params = new URLSearchParams();
    if (transport !== "all") {
        params.set("transport_kind", transport);
    }
    if (success !== null) {
        params.set("success", String(success));
    }
    const query = params.toString();
    const search = query ? `?${query}` : "";
    return apiFetch(`/admin/llm-calls${search}`, {
        schema: LLMCallListResponseSchema,
        signal,
    });
}

export function useLLMCalls(
    transport: TransportFilter,
    success: boolean | null,
): UseQueryResult<LLMCallListResponse, Error> {
    return useQuery({
        queryKey: llmCallKeys.list(transport, success),
        queryFn: ({ signal }) => fetchLLMCalls(transport, success, signal),
    });
}

async function fetchLLMCallStats(
    windowDays: number,
    transport: TransportFilter,
    signal: AbortSignal,
): Promise<LLMCallStats> {
    const params = new URLSearchParams({ window_days: String(windowDays) });
    if (transport !== "all") {
        params.set("transport_kind", transport);
    }
    return apiFetch(`/admin/llm-calls/stats?${params.toString()}`, {
        schema: LLMCallStatsSchema,
        signal,
    });
}

export function useLLMCallStats(
    windowDays: number,
    transport: TransportFilter,
): UseQueryResult<LLMCallStats, Error> {
    return useQuery({
        queryKey: llmCallKeys.stats(windowDays, transport),
        queryFn: ({ signal }) => fetchLLMCallStats(windowDays, transport, signal),
    });
}
