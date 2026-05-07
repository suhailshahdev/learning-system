/**
 * Session API client.
 *
 * Mirrors the backend's session_api.py and parsed_response.py
 * schemas. Mutation hooks for the four endpoints: start, sendTurn,
 * approve, abandon. No GET-by-ID hook yet because the backend has
 * no such endpoint. Add when in future needs reload-during-session
 * recovery.
 */
import {
    useMutation,
    useQuery,
    type UseMutationResult,
    type UseQueryResult,
} from "@tanstack/react-query";
import { z } from "zod";

import { apiFetch } from "@/lib/api/client";

// Enum schemas mirror app/models/enums.py exactly. Zod's z.enum
// gives both runtime validation and a TS literal union from the
// same declaration, keeping the contract single-sourced from the
// backend's StrEnum values.

export const DifficultySchema = z.enum(["beginner", "intermediate", "advanced"]);
export type Difficulty = z.infer<typeof DifficultySchema>;

export const LearningModeSchema = z.enum([
    "flashcard",
    "type_the_answer",
    "code_with_explanation",
    "multiple_choice",
    "explain_back",
    "socratic",
]);
export type LearningMode = z.infer<typeof LearningModeSchema>;

export const GradingVerdictSchema = z.enum([
    "correct",
    "partial",
    "incorrect",
    "open_graded",
]);
export type GradingVerdict = z.infer<typeof GradingVerdictSchema>;

export const SessionStateSchema = z.enum([
    "in_progress",
    "completed",
    "abandoned",
    "archived",
]);
export type SessionState = z.infer<typeof SessionStateSchema>;

export const TransportKindSchema = z.enum(["claude_playwright", "deepseek"]);
export type TransportKind = z.infer<typeof TransportKindSchema>;

// Parsed response shapes mirror app/schemas/parsed_response.py.
// The discriminated union on `kind` matches the backend's literal
// strings exactly so consumers can pattern-match in switch/case.

export const PrerequisiteSchema = z.object({
    topic_path: z.string().min(1),
    min_difficulty: DifficultySchema,
});
export type Prerequisite = z.infer<typeof PrerequisiteSchema>;

export const CodeBlockSchema = z.object({
    language: z.string().min(1),
    body: z.string().min(1),
});
export type CodeBlock = z.infer<typeof CodeBlockSchema>;

export const ParsedTurnSchema = z.object({
    kind: z.literal("turn"),
    topic_path: z.string().min(1),
    difficulty: DifficultySchema,
    prerequisites: z.array(PrerequisiteSchema),
    mode: LearningModeSchema,
    grading_verdict: GradingVerdictSchema.nullable(),
    grading_explanation: z.string().nullable(),
    grading_explanation_code: CodeBlockSchema.nullable(),
    question: z.string().min(1),
    question_code: CodeBlockSchema.nullable(),
    expected_answer: z.string().nullable(),
    requirements: z.string().nullable(),
    followup: z.string().nullable(),
    tags: z.array(z.string()),
});
export type ParsedTurn = z.infer<typeof ParsedTurnSchema>;

export const ParsedSessionEndSchema = z.object({
    kind: z.literal("session_end"),
    summary: z.string().min(1),
});
export type ParsedSessionEnd = z.infer<typeof ParsedSessionEndSchema>;

export const ParsedHandoverSchema = z.object({
    kind: z.literal("handover"),
    domain_focus: z.string(),
    covered: z.string(),
    last_question: z.string(),
    next_planned: z.string(),
    open_threads: z.string(),
    user_state: z.string(),
});
export type ParsedHandover = z.infer<typeof ParsedHandoverSchema>;

export const ParsedResponseSchema = z.discriminatedUnion("kind", [
    ParsedTurnSchema,
    ParsedSessionEndSchema,
    ParsedHandoverSchema,
]);
export type ParsedResponse = z.infer<typeof ParsedResponseSchema>;

// Session and request/response shapes mirror app/schemas/session_api.py.
// Datetimes come over the wire as ISO 8601 strings. Zod's z.string()
// keeps them as strings rather than parsing into Date because the
// frontend doesn't currently do date arithmetic. Promote to z.coerce.date()
// when a consumer needs Date semantics.

export const SessionResponseSchema = z.object({
    id: z.string(),
    topic_id: z.string().nullable(),
    state: SessionStateSchema,
    transport_kind: TransportKindSchema,
    mode_used: LearningModeSchema,
    claude_chat_url: z.string().nullable(),
    claude_chat_message_count: z.number().int().nonnegative(),
    created_at: z.string(),
    updated_at: z.string(),
});
export type SessionResponse = z.infer<typeof SessionResponseSchema>;

export const StartSessionResponseSchema = z.object({
    session: SessionResponseSchema,
    first_turn: ParsedTurnSchema,
});
export type StartSessionResponse = z.infer<typeof StartSessionResponseSchema>;

export const SendTurnResponseSchema = z.object({
    parsed: ParsedResponseSchema,
});
export type SendTurnResponse = z.infer<typeof SendTurnResponseSchema>;

export type StartSessionVariables = {
    topic_path: string;
    transport_kind: TransportKind;
};

export type SendTurnVariables = {
    session_id: string;
    answer: string;
};

export type SessionIdVariables = {
    session_id: string;
};

// Query key factory follows the precedent set by healthKeys. Even
// without queries today, the factory is the right place for keys
// once GET-by-ID lands in future.

export const sessionKeys = {
    all: () => ["sessions"] as const,
    detail: (id: string) => ["sessions", id] as const,
};

async function startSession(variables: StartSessionVariables): Promise<StartSessionResponse> {
    return apiFetch("/sessions", {
        method: "POST",
        body: variables,
        schema: StartSessionResponseSchema,
    });
}

async function sendTurn(variables: SendTurnVariables): Promise<SendTurnResponse> {
    return apiFetch(`/sessions/${variables.session_id}/turns`, {
        method: "POST",
        body: { answer: variables.answer },
        schema: SendTurnResponseSchema,
    });
}

async function approveSession(variables: SessionIdVariables): Promise<SessionResponse> {
    return apiFetch(`/sessions/${variables.session_id}/approve`, {
        method: "POST",
        schema: SessionResponseSchema,
    });
}

async function abandonSession(variables: SessionIdVariables): Promise<SessionResponse> {
    return apiFetch(`/sessions/${variables.session_id}/abandon`, {
        method: "POST",
        schema: SessionResponseSchema,
    });
}

export function useStartSession(): UseMutationResult<
    StartSessionResponse,
    Error,
    StartSessionVariables
> {
    return useMutation({
        mutationFn: startSession,
    });
}

export function useSendTurn(): UseMutationResult<
    SendTurnResponse,
    Error,
    SendTurnVariables
> {
    return useMutation({
        mutationFn: sendTurn,
    });
}

export function useApproveSession(): UseMutationResult<
    SessionResponse,
    Error,
    SessionIdVariables
> {
    return useMutation({
        mutationFn: approveSession,
    });
}

export function useAbandonSession(): UseMutationResult<
    SessionResponse,
    Error,
    SessionIdVariables
> {
    return useMutation({
        mutationFn: abandonSession,
    });
}

// Resume (cold-load) hook for GET /sessions/{id}.
//
// Returns the session row plus the latest parsed response so
// Session.tsx can render TurnView or SessionEndView without
// route state. Used as fallback when location.state is missing
// (deep links, refreshes, home dashboard clicks).

export const ResumeSessionResponseSchema = z.object({
    session: SessionResponseSchema,
    parsed: ParsedResponseSchema,
});
export type ResumeSessionResponse = z.infer<typeof ResumeSessionResponseSchema>;

async function fetchSession(
    sessionId: string,
    signal: AbortSignal,
): Promise<ResumeSessionResponse> {
    return apiFetch(`/sessions/${sessionId}`, {
        schema: ResumeSessionResponseSchema,
        signal,
    });
}

export function useSession(
    sessionId: string | undefined,
): UseQueryResult<ResumeSessionResponse, Error> {
    return useQuery({
        queryKey: sessionKeys.detail(sessionId ?? ""),
        queryFn: ({ signal }) => {
            if (sessionId === undefined) {
                throw new Error("session id is required");
            }
            return fetchSession(sessionId, signal);
        },
        enabled: sessionId !== undefined,
    });
}
