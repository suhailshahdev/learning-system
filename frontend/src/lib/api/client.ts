/**
 * Low-level API client and error types
 *
 * `apiFetch` performs an HTTP request against the backend and returns
 * parsed JSON. Failures surface as typed `ApiError` instances so
 * callers can discriminate between network, HTTP and parse errors
 * without string-matching
 */

import type { z } from "zod";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

/**
 * Kinds of failure `apiFetch` can produce.
 *
 * - `network`: could not reach the backend (DNS, offline, refused)
 * - `http`: backend responded with a non-2xx status
 * - `parse`: response was received but did not match the schema.
 */
export type ApiErrorKind = "network" | "http" | "parse";

export class ApiError extends Error {
    readonly kind: ApiErrorKind;
    readonly status: number | undefined;
    override readonly cause: unknown;

    constructor(kind: ApiErrorKind, message: string, options?: {status?: number; cause?: unknown}){
        super(message);
        this.name = "ApiError";
        this.kind = kind;
        this.status = options?.status;
        this.cause = options?.cause;
    }
}

type ApiFetchOptions<T> = {
    method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
    body?: unknown;
    schema: z.ZodType<T>;
    signal?: AbortSignal;
}

/**
 * Fetch a typed response from backend.
 *
 * Always validates the response against the given Zod schema; skipping
 * this is not an option because untyped response defeat the purpose
 * of having type client. Callers that genuinely do not care about
 * the response shape (rare) should pass `z.unknown()` explicitly.
 */
export async function apiFetch<T> (path: string, options:ApiFetchOptions<T>): Promise<T> {
    const url = `${API_BASE_URL}${path}`;
    const { method = "GET", body, schema, signal } = options;

    const init: RequestInit = { method };
    if (body !== undefined) {
        init.headers = { "Content-Type": "application/json"};
        init.body = JSON.stringify(body);
    }
    if (signal !== undefined) {
        init.signal = signal;
    }

    let response: Response;
    try {
        response = await fetch(url, init)
    } catch (error){
        // `fetch` only throws on network failures; HTTP errors resolve normally.
        throw new ApiError("network", `Network error reaching ${url}`, { cause: error });
    }

    if(!response.ok){
        throw new ApiError("http", `HTTP ${response.status} from ${url}`, {
            status: response.status
        });
    }

    let json: unknown;
    try {
        json = await response.json()
    } catch (error) {
        throw new ApiError("parse", `Response from ${url} was not valid JSON`, { cause: error });
    }

    const parsed = schema.safeParse(json);
    if (!parsed.success) {
        throw new ApiError("parse", `Response from ${url} did not match schema`, {
            cause: parsed.error
        })
    }

    return parsed.data
}
