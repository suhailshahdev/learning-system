import { useState } from "react";
import { Link, useLocation, useParams } from "react-router";
import { z } from "zod";

import {
    ParsedTurnSchema,
    SessionResponseSchema,
    type ParsedResponse,
    type ParsedTurn,
    type SessionResponse,
} from "@/lib/api";

const RouteStateSchema = z.object({
    firstTurn: ParsedTurnSchema,
    session: SessionResponseSchema,
});

type ResolvedState =
    | { kind: "loaded"; session: SessionResponse; parsed: ParsedResponse }
    | { kind: "missing" };

function resolveState(routeState: unknown): ResolvedState {
    const parsed = RouteStateSchema.safeParse(routeState);
    if (!parsed.success) {
        return { kind: "missing" };
    }
    return {
        kind: "loaded",
        session: parsed.data.session,
        parsed: parsed.data.firstTurn,
    };
}

export function Session(): React.JSX.Element {
    const { id } = useParams<{ id: string }>();
    const location = useLocation();
    const initial = resolveState(location.state);

    // Local state seeds from route state.
    const [parsed] = useState<ParsedResponse | null>(
        initial.kind === "loaded" ? initial.parsed : null,
    );

    if (initial.kind === "missing" || parsed === null) {
        return (
            <div className="min-h-svh bg-background text-foreground p-8">
                <p className="text-muted-foreground">
                    Session not loaded. Sessions can only be opened by starting them from home.
                </p>
                <Link
                    to="/"
                    className="text-sm underline underline-offset-4"
                >
                    Back to home
                </Link>
            </div>
        );
    }

    if (parsed.kind !== "turn") {
        return (
            <div className="min-h-svh bg-background text-foreground p-8">
                <p className="text-muted-foreground">
                    Unexpected response kind: {parsed.kind}. Session ID: {id}.
                </p>
            </div>
        );
    }

    return (
        <div className="min-h-svh bg-background text-foreground">
            <header className="flex items-center justify-between gap-4 p-4">
                <p className="text-sm text-muted-foreground">
                    Session {id}
                </p>
            </header>
            <main className="mx-auto flex w-full max-w-2xl flex-col gap-6 p-8">
                <ParsedTurnView turn={parsed} />
            </main>
        </div>
    );
}

function ParsedTurnView({ turn }: { turn: ParsedTurn }): React.JSX.Element {
    return (
        <article className="flex flex-col gap-4">
            <header className="flex flex-col gap-1">
                <p className="text-xs uppercase tracking-wide text-muted-foreground">
                    {turn.topic_path}
                </p>
                <p className="text-xs text-muted-foreground">
                    {turn.mode} · {turn.difficulty}
                </p>
            </header>
            <div className="text-base leading-relaxed whitespace-pre-wrap">
                {turn.question}
            </div>
            {turn.requirements !== null ? (
                <div className="rounded-md border border-border bg-muted/40 p-3 text-sm">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
                        Requirements
                    </p>
                    <p>{turn.requirements}</p>
                </div>
            ) : null}
        </article>
    );
}
