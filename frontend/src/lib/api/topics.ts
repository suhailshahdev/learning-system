/**
 * Topics view API client.
 *
 * Mirrors backend/app/schemas/topics.py. One query hook for the
 * GET /api/topics endpoint that returns flat domains and flat
 * topics. Frontend assembles the nested tree from parent_id.
 */
import { queryOptions, useQuery, type UseQueryResult } from "@tanstack/react-query";
import { z } from "zod";

import { apiFetch } from "@/lib/api/client";
import { TopicSummarySchema } from "@/lib/api/home";

const DomainKindSchema = z.enum([
    "language",
    "framework",
    "library",
    "concept",
    "tool",
    "practice",
    "other",
]);
export type DomainKind = z.infer<typeof DomainKindSchema>;

export const DomainSummarySchema = z.object({
    name: z.string(),
    kind: DomainKindSchema,
    description: z.string().nullable(),
});
export type DomainSummary = z.infer<typeof DomainSummarySchema>;

export const TopicsResponseSchema = z.object({
    domains: z.array(DomainSummarySchema),
    topics: z.array(TopicSummarySchema),
});
export type TopicsResponse = z.infer<typeof TopicsResponseSchema>;

export const topicsKeys = {
    all: () => ["topics"] as const,
};

async function fetchTopics(signal: AbortSignal): Promise<TopicsResponse> {
    return apiFetch("/topics", { schema: TopicsResponseSchema, signal });
}

export const topicsQueryOptions = queryOptions({
    queryKey: topicsKeys.all(),
    queryFn: ({ signal }) => fetchTopics(signal),
});

export function useTopics(): UseQueryResult<TopicsResponse, Error> {
    return useQuery(topicsQueryOptions);
}
