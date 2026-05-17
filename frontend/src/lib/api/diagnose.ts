/**
 * Diagnose API client.
 *
 * Mirrors the backend's diagnose_api.py schemas. One endpoint:
 * POST /diagnose returns a topic proposal with reasoning, or
 * 422 when there's nothing to diagnose (empty state).
 *
 * The 422 case is not a generic error: the request was well-formed
 * but the system has no diagnosable data yet. Consumers narrow
 * via ApiError.status to render an informational message instead
 * of an error.
 */
import {
  useMutation,
  type UseMutationResult,
} from "@tanstack/react-query";
import { z } from "zod";
import { apiFetch } from "@/lib/api/client";
import { TransportKindSchema } from "@/lib/api/sessions";

export const DiagnoseRequestSchema = z.object({
  transport_kind: TransportKindSchema,
});
export type DiagnoseRequest = z.infer<typeof DiagnoseRequestSchema>;

export const DiagnoseResponseSchema = z.object({
  topic_path: z.string().min(1),
  reasoning: z.string().min(1),
});
export type DiagnoseResponse = z.infer<typeof DiagnoseResponseSchema>;

async function diagnose(variables: DiagnoseRequest): Promise<DiagnoseResponse> {
  return apiFetch("/diagnose", {
    method: "POST",
    body: variables,
    schema: DiagnoseResponseSchema,
  });
}

export function useDiagnose(): UseMutationResult<
  DiagnoseResponse,
  Error,
  DiagnoseRequest
> {
  return useMutation({
    mutationFn: diagnose,
  });
}
