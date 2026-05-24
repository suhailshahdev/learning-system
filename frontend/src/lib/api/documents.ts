/**
 * Documents API client.
 *
 * Mirrors backend/app/schemas/document_api.py. POST /api/documents
 * ingests a pasted text document into the retrieval corpus: it gets
 * chunked, embedded, and stored. Ingest is user-triggered, so this
 * exposes a mutation.
 */
import { useMutation, type UseMutationResult } from "@tanstack/react-query";
import { z } from "zod";

import { apiFetch } from "@/lib/api/client";

export type IngestDocumentRequest = {
    title: string;
    content: string;
};

export const IngestDocumentResponseSchema = z.object({
    document_id: z.string(),
    title: z.string(),
    chunk_count: z.number(),
});
export type IngestDocumentResponse = z.infer<typeof IngestDocumentResponseSchema>;

async function ingestDocument(body: IngestDocumentRequest): Promise<IngestDocumentResponse> {
    return apiFetch("/documents", {
        method: "POST",
        body,
        schema: IngestDocumentResponseSchema,
    });
}

export function useIngestDocument(): UseMutationResult<
    IngestDocumentResponse,
    Error,
    IngestDocumentRequest
> {
    return useMutation({ mutationFn: ingestDocument });
}
