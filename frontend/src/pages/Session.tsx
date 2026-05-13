import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router";
import { z } from "zod";

import { EndSessionButton } from "@/components/session/end-session-button";
import { GradingView } from "@/components/session/grading-view";
import { SessionEndView } from "@/components/session/session-end-view";
import { TurnView } from "@/components/session/turn-view";
import {
    ParsedTurnSchema,
    SessionResponseSchema,
    useContinueSession,
    useSendTurn,
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

    // Local state holds the latest parsed response after the user
    // begins interacting. Null means render from route or remote
    // source. Updated after sendTurn and continueSession succeed.
    const [localParsed, setLocalParsed] = useState<ParsedResponse | null>(null);

    const sendTurn = useSendTurn();
    const continueSession = useContinueSession();

    // Prefetch the next teaching turn the moment a grading response
    // lands. The user reads the grading, the round trip happens in
    // parallel, the Continue button reads from the in-flight mutation
    // state. If the user clicks before the prefetch returns, the
    // pending state shows "Loading next question..." until it does.
    //
    // The prefetch survives navigation within this Session mount.
    // Tab close mid-prefetch wastes one LLM call on DeepSeek (billed)
    // or zero cost on Playwright (Page closes).
    //
    // useEffect rather than firing in sendTurn's onSuccess because
    // grading can also arrive on a cold-load (route.kind === "missing"
    // and the user refreshed mid-cycle). The effect covers both paths.
    //
    // The continue's success path is NOT handled via an effect: the
    // mutation's data is consulted directly in resolveCurrentParsed,
    // so the next turn renders as soon as the mutation resolves
    // without a setState round trip.
    useEffect(() => {
        if (id === undefined) {
            return;
        }
        const parsed = resolveCurrentParsed(
            localParsed,
            route,
            remoteSession.data?.parsed,
            continueSession.data?.parsed,
        );
        if (parsed?.kind !== "grading") {
            return;
        }
        if (continueSession.isIdle) {
            continueSession.mutate({ session_id: id });
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [id, localParsed, remoteSession.data, continueSession.data]);

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

    const parsed = resolveCurrentParsed(
        localParsed,
        route,
        remoteSession.data?.parsed,
        continueSession.data?.parsed,
    );

    if (parsed === null) {
        return <NotLoaded message="Session not loaded." />;
    }

    // Handover is unexpected at the frontend because the backend
    // handles chat transitions transparently inside request_next_question.
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

    const handleAnswerSubmit = (answer: string): void => {
        sendTurn.mutate(
            { session_id: id, answer },
            {
                onSuccess: (data) => {
                    setLocalParsed(data.parsed);
                },
            },
        );
    };

    const handleContinueClick = (): void => {
        // Prefetch may have already returned. The useSuccess effect
        // will pick it up. If still in-flight, the disabled state
        // on the Continue button keeps the user waiting. If reset
        // (rare: user re-clicks after success), fire a fresh mutation.
        if (continueSession.isIdle) {
            continueSession.mutate({ session_id: id });
        }
    };

    return (
        <div className="min-h-svh bg-background text-foreground">
            <header className="flex items-center justify-between gap-4 p-4">
                <p className="text-sm text-muted-foreground">
                    Session {id}
                </p>
                {parsed.kind === "turn" || parsed.kind === "grading" ? (
                    <EndSessionButton
                        sessionId={id}
                        onAbandoned={handleAbandoned}
                    />
                ) : null}
            </header>
            <main className="mx-auto flex w-full max-w-2xl flex-col gap-6 p-8">
                {parsed.kind === "turn" ? (
                    <TurnView
                        turn={parsed}
                        onSubmit={handleAnswerSubmit}
                        isSubmitting={sendTurn.isPending}
                        submitError={sendTurn.isError ? sendTurn.error.message : null}
                    />
                ) : parsed.kind === "grading" ? (
                    <GradingView
                        grading={parsed}
                        onContinue={handleContinueClick}
                        isContinuing={continueSession.isPending}
                        continueError={
                            continueSession.isError ? continueSession.error.message : null
                        }
                    />
                ) : (
                    <SessionEndView parsed={parsed} sessionId={id} />
                )}
            </main>
        </div>
    );
}

function resolveCurrentParsed(
    localParsed: ParsedResponse | null,
    route: RouteState,
    remoteParsed: ParsedResponse | undefined,
    continueParsed: ParsedResponse | undefined,
): ParsedResponse | null {
    // Priority: sendTurn's local result wins (most recent user
    // action). Then continueSession's prefetched result if it has
    // resolved. Then route state (warm load). Then remote query
    // data (cold load). Null when nothing is available yet.
    //
    // continueParsed sits between localParsed and route because it
    // represents data newer than the route's first turn but older
    // than any subsequent answer the user submits. Once the user
    // answers again, sendTurn writes localParsed and overrides.
    if (localParsed !== null) {
        return localParsed;
    }
    if (continueParsed !== undefined) {
        return continueParsed;
    }
    if (route.kind === "loaded") {
        return route.parsed;
    }
    return remoteParsed ?? null;
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
