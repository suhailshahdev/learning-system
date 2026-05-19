import { Link, useSearchParams } from "react-router";

import { ApiStatus } from "@/components/api-status";
import { ModeToggle } from "@/components/mode-toggle";
import { Card, CardContent } from "@/components/ui/card";
import {
    useBrowse,
    type BrowseSessionRow,
    type BrowseStateFilter,
} from "@/lib/api/browse";

// Tab values map to the BrowseStateFilter union. Order is the
// visual order on the page: "All" first, then the four states
// in the order most useful for retest-finding workflow.
const TABS: ReadonlyArray<{ value: BrowseStateFilter; label: string }> = [
    { value: "all", label: "All" },
    { value: "completed", label: "Completed" },
    { value: "in_progress", label: "In progress" },
    { value: "abandoned", label: "Abandoned" },
    { value: "archived", label: "Archived" },
];

const STATE_LABELS: Record<BrowseSessionRow["state"], string> = {
    in_progress: "in progress",
    completed: "completed",
    abandoned: "abandoned",
    archived: "archived",
};

const STATE_STYLES: Record<BrowseSessionRow["state"], string> = {
    in_progress: "bg-success/15 text-success",
    completed: "bg-muted text-muted-foreground",
    abandoned: "bg-destructive/15 text-destructive",
    archived: "bg-muted text-muted-foreground",
};

function parseStateFilter(raw: string | null): BrowseStateFilter {
    // Validate the URL value against the known set. Unknown or
    // missing values fall back to "all" rather than 422-ing the
    // request, since a deep-linked URL with a typo should still
    // load something useful.
    if (raw === null) {
        return "all";
    }
    const match = TABS.find((tab) => tab.value === raw);
    return match ? match.value : "all";
}

export function Browse(): React.JSX.Element {
    const [searchParams, setSearchParams] = useSearchParams();
    const state = parseStateFilter(searchParams.get("state"));
    const query = useBrowse(state);

    const handleTabClick = (next: BrowseStateFilter): void => {
        // "all" clears the param, everything else writes it.
        // Keeps the URL clean for the default view.
        if (next === "all") {
            setSearchParams({});
        } else {
            setSearchParams({ state: next });
        }
    };

    return (
        <div className="min-h-svh bg-background text-foreground">
            <header className="flex items-center justify-between gap-4 p-4">
                <nav className="flex items-center gap-4 text-sm">
                    <Link to="/" className="underline underline-offset-4">
                        Home
                    </Link>
                    <Link to="/topics" className="underline underline-offset-4">
                        Topics
                    </Link>
                </nav>
                <div className="flex items-center gap-4">
                    <ApiStatus />
                    <ModeToggle />
                </div>
            </header>
            <main className="p-8">
                <div className="mx-auto flex w-full max-w-2xl flex-col gap-4">
                    <h1 className="text-3xl font-bold">Sessions</h1>
                    <p className="text-sm text-muted-foreground">
                        Past and current learning sessions.
                    </p>
                    <TabRow active={state} onSelect={handleTabClick} />
                    <BrowseContent query={query} activeState={state} />
                </div>
            </main>
        </div>
    );
}

type TabRowProps = {
    active: BrowseStateFilter;
    onSelect: (next: BrowseStateFilter) => void;
};

function TabRow({ active, onSelect }: TabRowProps): React.JSX.Element {
    return (
        <div
            role="tablist"
            className="flex flex-wrap gap-2 border-b border-border pb-3"
        >
            {TABS.map((tab) => {
                const isActive = tab.value === active;
                return (
                    <button
                        key={tab.value}
                        type="button"
                        role="tab"
                        aria-selected={isActive}
                        onClick={() => { onSelect(tab.value); }}
                        className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
                            isActive
                                ? "bg-accent text-accent-foreground"
                                : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                        }`}
                    >
                        {tab.label}
                    </button>
                );
            })}
        </div>
    );
}

type BrowseContentProps = {
    query: ReturnType<typeof useBrowse>;
    activeState: BrowseStateFilter;
};

function BrowseContent({ query, activeState }: BrowseContentProps): React.JSX.Element {
    if (query.isPending) {
        return <p className="text-muted-foreground">Loading sessions...</p>;
    }
    if (query.isError) {
        return (
            <p className="text-destructive">
                Failed to load sessions: {query.error.message}
            </p>
        );
    }
    const { rows, limit_reached } = query.data;
    if (rows.length === 0) {
        return <EmptyState activeState={activeState} />;
    }
    return (
        <div className="flex flex-col gap-3">
            <ul className="flex flex-col gap-2">
                {rows.map((row) => (
                    <li key={row.id}>
                        <SessionRow row={row} />
                    </li>
                ))}
            </ul>
            {limit_reached ? (
                <p className="text-xs text-muted-foreground">
                    Showing the most recent 50 sessions. Older ones are not yet
                    browsable.
                </p>
            ) : null}
        </div>
    );
}

function EmptyState({ activeState }: { activeState: BrowseStateFilter }): React.JSX.Element {
    // Different copy depending on whether the user is filtering or
    // looking at a truly empty list. Filtering with no matches is
    // user-friendly to call out specifically.
    const message =
        activeState === "all"
            ? "No sessions yet. Start one from the home page."
            : `No ${activeState.replace("_", " ")} sessions.`;
    return (
        <Card>
            <CardContent className="p-6">
                <p className="text-sm text-muted-foreground">{message}</p>
            </CardContent>
        </Card>
    );
}

type SessionRowProps = {
    row: BrowseSessionRow;
};

function SessionRow({ row }: SessionRowProps): React.JSX.Element {
    const topicLabel = row.topic_path ?? "Unspecified topic";
    const stateLabel = STATE_LABELS[row.state];
    const stateClass = STATE_STYLES[row.state];

    const content = (
        <div className="flex items-start justify-between gap-3">
            <div className="flex flex-col gap-1">
                <p className="text-sm font-medium">{topicLabel}</p>
                <p className="text-xs text-muted-foreground">
                    {row.mode_used} · {row.transport_kind} · {row.learned_item_count}{" "}
                    {row.learned_item_count === 1 ? "item" : "items"}
                </p>
            </div>
            <span
                className={`shrink-0 rounded-md px-2 py-0.5 text-xs font-medium ${stateClass}`}
            >
                {stateLabel}
            </span>
        </div>
    );

    // Row routing mirrors RecentSessions:
    // - in_progress → live session page
    // - completed / abandoned → transcript
    // - archived → inert (no surface yet)
    if (row.state === "in_progress") {
        return (
            <Link
                to={`/session/${row.id}`}
                className="block rounded-md border border-border p-3 transition-colors hover:bg-accent"
            >
                {content}
            </Link>
        );
    }
    if (row.state === "completed" || row.state === "abandoned") {
        return (
            <Link
                to={`/session/${row.id}/transcript`}
                className="block rounded-md border border-border p-3 transition-colors hover:bg-accent"
            >
                {content}
            </Link>
        );
    }
    return <div className="rounded-md border border-border p-3">{content}</div>;
}
