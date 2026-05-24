/**
 * API client public surface.
 *
 * Consumers import from "@/lib/api", not from specific files.
 * Adding a new feature means creating `@/lib/api/<feature>.ts` and
 * re-exporting from here.
 */

export { ApiError } from "@/lib/api/client";
export type { ApiErrorKind } from "@/lib/api/client";
export * from "@/lib/api/browse";
export * from "@/lib/api/diagnose";
export * from "@/lib/api/documents";
export * from "@/lib/api/health";
export * from "@/lib/api/home";
export * from "@/lib/api/search";
export * from "@/lib/api/sessions";
export * from "@/lib/api/topics";
export * from "@/lib/api/transcript";
