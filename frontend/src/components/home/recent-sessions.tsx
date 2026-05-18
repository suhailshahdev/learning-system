import { Link } from "react-router";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { RecentSessionSummary } from "@/lib/api";

type RecentSessionsProps = {
    sessions: RecentSessionSummary[];
};

const STATE_LABELS: Record<RecentSessionSummary["state"], string> = {
    in_progress: "in progress",
    completed: "completed",
    abandoned: "abandoned",
    archived: "archived",
};

const STATE_STYLES: Record<RecentSessionSummary["state"], string> = {
    in_progress: "bg-success/15 text-success",
    completed: "bg-muted text-muted-foreground",
    abandoned: "bg-destructive/15 text-destructive",
    archived: "bg-muted text-muted-foreground",
};

export function RecentSessions({ sessions }: RecentSessionsProps): React.JSX.Element {
    return (
        <Card>
            <CardHeader>
                <CardTitle>Recent sessions</CardTitle>
                <CardDescription>
                    Your latest learning sessions, most recent first.
                </CardDescription>
            </CardHeader>
            <CardContent>
                {sessions.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                        No sessions yet. Start one above to begin.
                    </p>
                ) : (
                    <ul className="flex flex-col gap-3">
                        {sessions.map((session) => (
                            <li key={session.id}>
                                <SessionRow session={session} />
                            </li>
                        ))}
                    </ul>
                )}
            </CardContent>
        </Card>
    );
}

type SessionRowProps = {
    session: RecentSessionSummary;
};

function SessionRow({ session }: SessionRowProps): React.JSX.Element {
    const topicLabel = session.topic_path ?? "Unspecified topic";
    const stateLabel = STATE_LABELS[session.state];
    const stateClass = STATE_STYLES[session.state];

    const content = (
        <div className="flex items-start justify-between gap-3">
            <div className="flex flex-col gap-1">
                <p className="text-sm font-medium">{topicLabel}</p>
                <p className="text-xs text-muted-foreground">
                    {session.mode_used} · {session.transport_kind}
                </p>
            </div>
            <span
                className={`shrink-0 rounded-md px-2 py-0.5 text-xs font-medium ${stateClass}`}
            >
                {stateLabel}
            </span>
        </div>
    );

    // In-progress sessions route to the live session page.
    // Completed and abandoned sessions route to the read-only
    // transcript. Archived sessions stay inert until archive
    // browsing lands as its own surface.
    if (session.state === "in_progress") {
        return (
            <Link
                to={`/session/${session.id}`}
                className="block rounded-md p-2 transition-colors hover:bg-accent"
            >
                {content}
            </Link>
        );
    }

    if (session.state === "completed" || session.state === "abandoned") {
        return (
            <Link
                to={`/session/${session.id}/transcript`}
                className="block rounded-md p-2 transition-colors hover:bg-accent"
            >
                {content}
            </Link>
        );
    }

    return <div className="p-2">{content}</div>;
}
