import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router";
import { z } from "zod";

import { EndSessionButton } from "@/components/session/end-session-button";
import { GradingView } from "@/components/session/grading-view";
import { SessionEndView } from "@/components/session/session-end-view";
import { TurnView } from "@/components/session/turn-view";
import {
    ParsedTurnSchema,
    sessionKeys,
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
    const queryClient = useQueryClient();

    // Prefetch the next teaching turn the moment a grading response
    // lands. The user reads the grading, the round trip happens in
    // parallel, the Continue click advances display to the prefetched
    // turn instantly. If the user clicks before the prefetch returns,
    // the click clears localParsed and the loading fallback shows
    // until the mutation resolves.
    //
    // The prefetch fires from this effect (not from sendTurn's
    // onSuccess) because grading can also arrive on a cold-load when
    // the user refreshed mid-cycle. The effect covers both paths
    // (live answer submission and cold-load resume).
    //
    // For the effect to fire on each new cycle, the previous cycle's
    // continueSession state must be reset before localParsed gets the
    // new grading. handleAnswerSubmit does that reset.
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
            continueSession.mutate(
                { session_id: id },
                {
                    onSuccess: () => {
                        // Server state advanced. Invalidate resume cache
                        // so re-entry to the URL fetches fresh state.
                        void queryClient.invalidateQueries({
                            queryKey: sessionKeys.detail(id),
                        });
                    },
                },
            );
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
                    // Reset the previous cycle's continueSession state
                    // before storing the new grading. Without this reset,
                    // the useEffect that auto-fires the prefetch sees
                    // continueSession.isIdle === false (carry-over from
                    // last cycle's success), and the next cycle never
                    // prefetches.
                    continueSession.reset();
                    setLocalParsed(data.parsed);
                    // Server state advanced. Invalidate the cached
                    // resume query so a future cold-load fetches fresh
                    // rather than returning a stale earlier turn.
                    void queryClient.invalidateQueries({
                        queryKey: sessionKeys.detail(id),
                    });
                },
            },
        );
    };

    const handleContinueClick = (): void => {
        // Advance display past the grading. Three cases:
        //
        // - Prefetch resolved: clearing localParsed lets the resolution
        //   priority hand off to continueParsed (the next teaching turn).
        // - Prefetch pending: clearing localParsed shows the loading
        //   fallback until the mutation resolves. The button's disabled
        //   state already prevents this click in practice, but the
        //   reset makes the behavior consistent if the click slips
        //   through (rapid double-click, etc.).
        // - Prefetch idle: shouldn't happen since the useEffect fires
        //   it on grading arrival, but defensive: fire and clear.
        if (continueSession.isIdle) {
            continueSession.mutate({ session_id: id });
        }
        setLocalParsed(null);
        // Server state advanced (or will, once the mutation resolves).
        // Invalidate the resume cache so re-entry to the URL fetches
        // fresh state.
        void queryClient.invalidateQueries({
            queryKey: sessionKeys.detail(id),
        });
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
