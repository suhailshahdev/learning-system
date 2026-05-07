import { useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router";
import { z } from "zod";

import { EndSessionButton } from "@/components/session/end-session-button";
import { SessionEndView } from "@/components/session/session-end-view";
import { TurnView } from "@/components/session/turn-view";
import {
    ParsedTurnSchema,
    SessionResponseSchema,
    useSession,
    type ParsedResponse,
} from "@/lib/api";

const RouteStateSchema = z.object({
    firstTurn: ParsedTurnSchema,
    session: SessionResponseSchema,
});

type RouteState =
    | { kind: "loaded"; parsed: ParsedResponse }
    | { kind: "missing" };

function resolveRouteState(routeState: unknown): RouteState {
    const parsed = RouteStateSchema.safeParse(routeState);
    if (!parsed.success) {
        return { kind: "missing" };
    }
    return { kind: "loaded", parsed: parsed.data.firstTurn };
}

export function Session(): React.JSX.Element {
    const { id } = useParams<{ id: string }>();
    const location = useLocation();
    const navigate = useNavigate();

    const route = resolveRouteState(location.state);
    const remoteSession = useSession(route.kind === "missing" ? id : undefined);

    // Local state holds turn updates after the user starts answering.
    // Null means "no local update yet, render from route or remote
    // source." setLocalParsed is called from handleResponse only.
    const [localParsed, setLocalParsed] = useState<ParsedResponse | null>(null);

    // Bumps on every onResponse so TurnView remounts and clears its
    // internal answer state. Without the key, React reuses the same
    // instance across prop changes and the previous answer would
    // persist in the textarea.
    const [turnIndex, setTurnIndex] = useState(0);

    const handleResponse = (next: ParsedResponse): void => {
        setLocalParsed(next);
        setTurnIndex((n) => n + 1);
    };

    if (id === undefined) {
        return <NotLoaded message="No session id in the URL." />;
    }

    if (route.kind === "missing" && remoteSession.isPending) {
        return (
            <div className="min-h-svh bg-background text-foreground p-8">
                <p className="text-muted-foreground">Loading session...</p>
            </div>
        );
    }

    if (route.kind === "missing" && remoteSession.isError) {
        return <NotLoaded message={remoteSession.error.message} />;
    }

    // Resolve which parsed response to render. Local turn updates take
    // priority once the user has answered at least once. Otherwise fall
    // back to route state on a warm load or query data on a cold load.
    const parsed: ParsedResponse | null =
        localParsed
        ?? (route.kind === "loaded" ? route.parsed : (remoteSession.data?.parsed ?? null));

    if (parsed === null) {
        return <NotLoaded message="Session not loaded." />;
    }

    // Handover is unexpected at the frontend because the backend
    // handles chat transitions transparently inside send_user_answer.
    if (parsed.kind === "handover") {
        return (
            <div className="min-h-svh bg-background text-foreground p-8">
                <p className="text-muted-foreground">
                    Unexpected response kind: {parsed.kind}. Session ID: {id}.
                </p>
            </div>
        );
    }

    const handleAbandoned = (): void => {
        void navigate("/");
    };

    return (
        <div className="min-h-svh bg-background text-foreground">
            <header className="flex items-center justify-between gap-4 p-4">
                <p className="text-sm text-muted-foreground">
                    Session {id}
                </p>
                {parsed.kind === "turn" ? (
                    <EndSessionButton
                        sessionId={id}
                        onAbandoned={handleAbandoned}
                    />
                ) : null}
            </header>
            <main className="mx-auto flex w-full max-w-2xl flex-col gap-6 p-8">
                {parsed.kind === "turn" ? (
                    <TurnView
                        key={turnIndex}
                        turn={parsed}
                        sessionId={id}
                        onResponse={handleResponse}
                    />
                ) : (
                    <SessionEndView parsed={parsed} sessionId={id} />
                )}
            </main>
        </div>
    );
}

function NotLoaded({ message }: { message: string }): React.JSX.Element {
    return (
        <div className="min-h-svh bg-background text-foreground p-8">
            <p className="text-muted-foreground">{message}</p>
            <Link to="/" className="text-sm underline underline-offset-4">
                Back to home
            </Link>
        </div>
    );
}
