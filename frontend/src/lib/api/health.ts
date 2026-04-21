/**
 * Health check queries.
 * 
 * Thin wrapper around GET /api/health: schemas that mirror the
 * backend's HealthReponse, plus a pre-configured query for use with
 * TanStack Query. The backend returns 503 for the degraded case; that
 * surfaces as an ApiError with kind "http" and status 503
 */

import { queryOptions, useQuery, type UseQueryResult } from "@tanstack/react-query";
import { z } from "zod";

import { apiFetch } from "@/lib/api/client";

const ComponentHealthSchema = z.object({
    status: z.enum(["ok", "error"]),
    detail: z.string().nullable
});

export const HealthResponseSchema = z.object({
    status: z.enum(["ok", "degraded"]),
    components: z.record(z.string(), ComponentHealthSchema)
});

export type HealthResponse = z.infer<typeof HealthResponseSchema>;

export const healthKeys = {
    all: () => ["health"] as const
}

async function fetchHealth(signal: AbortSignal): Promise<HealthResponse> {
    return apiFetch("/health", { schema: HealthResponseSchema, signal });
}

export const healthQueryOptions = queryOptions({
    queryKey: healthKeys.all(),
    queryFn: ({signal}) => fetchHealth(signal),
    // Health should reflect reality promptly; 10s is tigther than the
    // global 30s default
    staleTime: 10_000,
});

export function useHealth(): UseQueryResult<HealthResponse, Error> {
    return useQuery(healthQueryOptions)
}