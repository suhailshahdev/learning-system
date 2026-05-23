/**
 * Search API client.
 *
 * Mirrors backend/app/schemas/search_api.py. POST /api/search embeds
 * a query and returns ranked corpus hits. Search is user-triggered,
 * not load-time, so this exposes a mutation rather than a query.
 */
import { useMutation, type UseMutationResult } from "@tanstack/react-query";
import { z } from "zod";

import { apiFetch } from "@/lib/api/client";

const EmbeddingSourceTypeSchema = z.enum(["learned_item", "document_chunk"]);
export type EmbeddingSourceType = z.infer<typeof EmbeddingSourceTypeSchema>;

export const SearchHitSchema = z.object({
    source_type: EmbeddingSourceTypeSchema,
    source_id: z.string(),
    content: z.string(),
    score: z.number(),
});
export type SearchHit = z.infer<typeof SearchHitSchema>;

export const SearchResponseSchema = z.object({
    query: z.string(),
    hits: z.array(SearchHitSchema),
});
export type SearchResponse = z.infer<typeof SearchResponseSchema>;

export type SearchRequest = {
    query: string;
    limit?: number;
};

async function postSearch(body: SearchRequest): Promise<SearchResponse> {
    return apiFetch("/search", {
        method: "POST",
        body,
        schema: SearchResponseSchema,
    });
}

export function useSearch(): UseMutationResult<SearchResponse, Error, SearchRequest> {
    return useMutation({ mutationFn: postSearch });
}
