/**
 * Home dashboard API client.
 *
 * Mirrors the backend home schemas. One query hook for GET /api/home.
 * Datetime fields stay as ISO strings for now and can be promoted
 * to parsed dates when relative-time formatting is added.
 */
import { queryOptions, useQuery, type UseQueryResult } from "@tanstack/react-query";
import { z } from "zod";

import { apiFetch } from "@/lib/api/client";
import {
    DifficultySchema,
    LearningModeSchema,
} from "@/lib/api/sessions";

// TopicStatus mirrors app/models/enums.py. Local to home.ts because
// no other client module needs it today.
const TopicStatusSchema = z.enum([
    "not_started",
    "in_progress",
    "learned",
    "needs_revision",
]);
export type TopicStatus = z.infer<typeof TopicStatusSchema>;

export const TopicSummarySchema = z.object({
    id: z.string(),
    parent_id: z.string().nullable(),
    path: z.string(),
    name: z.string(),
    domain: z.string(),
    difficulty: DifficultySchema.nullable(),
    status: TopicStatusSchema,
});
export type TopicSummary = z.infer<typeof TopicSummarySchema>;

export const DomainFocusSchema = z.object({
    domain: z.string(),
    in_progress_topics: z.array(TopicSummarySchema),
});
export type DomainFocus = z.infer<typeof DomainFocusSchema>;

export const LearnedItemSummarySchema = z.object({
    id: z.string(),
    question: z.string(),
    topic_path: z.string(),
    difficulty: DifficultySchema.nullable(),
    mode: LearningModeSchema,
    last_reviewed_at: z.string().nullable(),
});
export type LearnedItemSummary = z.infer<typeof LearnedItemSummarySchema>;

export const KnowledgeSummaryRowSchema = z.object({
    domain: z.string(),
    difficulty: DifficultySchema,
    count: z.number().int().nonnegative(),
});
export type KnowledgeSummaryRow = z.infer<typeof KnowledgeSummaryRowSchema>;

// RecentSessionSummary mirrors the backend schema. Drops claude_chat_url
// and claude_chat_message_count (internal to the live session loop)
// and adds topic_path joined from the Topic table.
export const RecentSessionSummarySchema = z.object({
    id: z.string(),
    topic_id: z.string().nullable(),
    topic_path: z.string().nullable(),
    state: z.enum(["in_progress", "completed", "abandoned", "archived"]),
    transport_kind: z.enum(["claude_playwright", "deepseek"]),
    mode_used: LearningModeSchema,
    created_at: z.string(),
    updated_at: z.string(),
});
export type RecentSessionSummary = z.infer<typeof RecentSessionSummarySchema>;

export const HomeResponseSchema = z.object({
    is_blank_slate: z.boolean(),
    continue_last: RecentSessionSummarySchema.nullable(),
    due_for_review: z.array(LearnedItemSummarySchema),
    focus_by_domain: z.array(DomainFocusSchema),
    recent_sessions: z.array(RecentSessionSummarySchema),
    knowledge_summary: z.array(KnowledgeSummaryRowSchema),
});
export type HomeResponse = z.infer<typeof HomeResponseSchema>;

export const homeKeys = {
    all: () => ["home"] as const,
};

async function fetchHome(signal: AbortSignal): Promise<HomeResponse> {
    return apiFetch("/home", { schema: HomeResponseSchema, signal });
}

export const homeQueryOptions = queryOptions({
    queryKey: homeKeys.all(),
    queryFn: ({ signal }) => fetchHome(signal),
});

export function useHome(): UseQueryResult<HomeResponse, Error> {
    return useQuery(homeQueryOptions);
}
