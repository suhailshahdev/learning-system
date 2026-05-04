import { useState } from "react";
import { Link, useLocation, useParams } from "react-router";
import { z } from "zod";

import { TurnView } from "@/components/session/turn-view";
import {
    ParsedTurnSchema,
    SessionResponseSchema,
    type ParsedResponse,
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

    const [parsed, setParsed] = useState<ParsedResponse | null>(
        initial.kind === "loaded" ? initial.parsed : null,
    );
    // Bumps on every onResponse so TurnView remounts and clears its
    // internal answer state. Without the key, React reuses the same
    // instance across prop changes and the previous answer would
    // persist in the textarea.
    const [turnIndex, setTurnIndex] = useState(0);

    const handleResponse = (next: ParsedResponse): void => {
        setParsed(next);
        setTurnIndex((n) => n + 1);
    };

    if (initial.kind === "missing" || parsed === null || id === undefined) {
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
        // Handover is treated as unexpected because the
        // backend transparently handles chat transitions
        // inside send_user_answer.
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
                <TurnView
                    key={turnIndex}
                    turn={parsed}
                    sessionId={id}
                    onResponse={handleResponse}
                />
            </main>
        </div>
    );
}
