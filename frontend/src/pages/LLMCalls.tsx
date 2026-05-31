import { useSearchParams } from "react-router";
import { Link } from "react-router";

import { ApiStatus } from "@/components/api-status";
import { ModeToggle } from "@/components/mode-toggle";
import { Card, CardContent } from "@/components/ui/card";
import {
    useLLMCalls,
    useLLMCallStats,
    type LLMCallRow,
    type TransportFilter,
} from "@/lib/api/llm-calls";

// Window for the stats panel, in days. Fixed for now, a selector
// is a later refinement. Matches the backend's default window.
const STATS_WINDOW_DAYS = 7;

// Transport filter tabs. "all" first, then the two transports.
// Values map to the TransportFilter union, the labels are the
// human-readable transport names.
const TRANSPORT_TABS: ReadonlyArray<{ value: TransportFilter; label: string }> = [
    { value: "all", label: "All" },
    { value: "claude_playwright", label: "Claude" },
    { value: "deepseek", label: "DeepSeek" },
];

// Success filter tabs. The URL stores "true"/"false"/absent, the
// page maps absent to "all calls".
type SuccessFilter = "all" | "ok" | "errors";
const SUCCESS_TABS: ReadonlyArray<{ value: SuccessFilter; label: string }> = [
    { value: "all", label: "All" },
    { value: "ok", label: "Succeeded" },
    { value: "errors", label: "Failed" },
];

function parseTransport(raw: string | null): TransportFilter {
    if (raw === null) {
        return "all";
    }
    const match = TRANSPORT_TABS.find((tab) => tab.value === raw);
    return match ? match.value : "all";
}

function parseSuccess(raw: string | null): SuccessFilter {
    const match = SUCCESS_TABS.find((tab) => tab.value === raw);
    return match ? match.value : "all";
}

// Maps the success tab to the backend's boolean | null filter.
function successToBool(filter: SuccessFilter): boolean | null {
    if (filter === "ok") {
        return true;
    }
    if (filter === "errors") {
        return false;
    }
    return null;
}

export function LLMCalls(): React.JSX.Element {
    const [searchParams, setSearchParams] = useSearchParams();
    const transport = parseTransport(searchParams.get("transport"));
    const success = parseSuccess(searchParams.get("success"));

    const statsQuery = useLLMCallStats(STATS_WINDOW_DAYS, transport);
    const listQuery = useLLMCalls(transport, successToBool(success));

    const setParam = (key: string, value: string | null): void => {
        const next = new URLSearchParams(searchParams);
        if (value === null) {
            next.delete(key);
        } else {
            next.set(key, value);
        }
        setSearchParams(next);
    };

    return (
        <div className="min-h-svh bg-background text-foreground">
            <header className="flex items-center justify-between gap-4 p-4">
                <nav className="flex items-center gap-4 text-sm">
                    <Link to="/" className="underline underline-offset-4">
                        Home
                    </Link>
                    <Link to="/sessions" className="underline underline-offset-4">
                        Sessions
                    </Link>
                </nav>
                <div className="flex items-center gap-4">
                    <ApiStatus />
                    <ModeToggle />
                </div>
            </header>
            <main className="p-8">
                <div className="mx-auto flex w-full max-w-4xl flex-col gap-6">
                    <div className="flex flex-col gap-1">
                        <h1 className="text-3xl font-bold">LLM calls</h1>
                        <p className="text-sm text-muted-foreground">
                            Recorded transport round-trips and aggregate stats.
                        </p>
                    </div>

                    <StatsPanel query={statsQuery} transport={transport} />

                    <div className="flex flex-col gap-3">
                        <div className="flex flex-wrap items-center gap-4">
                            <FilterRow
                                label="Transport"
                                tabs={TRANSPORT_TABS}
                                active={transport}
                                onSelect={(v) => {
                                    setParam("transport", v === "all" ? null : v);
                                }}
                            />
                            <FilterRow
                                label="Status"
                                tabs={SUCCESS_TABS}
                                active={success}
                                onSelect={(v) => {
                                    setParam("success", v === "all" ? null : v);
                                }}
                            />
                        </div>
                        <CallList query={listQuery} />
                    </div>
                </div>
            </main>
        </div>
    );
}

type StatsPanelProps = {
    query: ReturnType<typeof useLLMCallStats>;
    transport: TransportFilter;
};

function StatsPanel({ query, transport }: StatsPanelProps): React.JSX.Element {
    if (query.isPending) {
        return <p className="text-muted-foreground">Loading stats...</p>;
    }
    if (query.isError) {
        return (
            <p className="text-destructive">
                Failed to load stats: {query.error.message}
            </p>
        );
    }
    const stats = query.data;
    const scope = transport === "all" ? "all transports" : transport;
    return (
        <Card>
            <CardContent className="flex flex-col gap-4 p-6">
                <p className="text-xs text-muted-foreground">
                    Last {stats.window_days} days, {scope}
                </p>
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
                    <Stat label="Calls" value={String(stats.total_calls)} />
                    <Stat label="Errors" value={String(stats.error_count)} />
                    <Stat
                        label="Error rate"
                        value={formatRate(stats.error_rate)}
                    />
                    <Stat
                        label="Latency p50"
                        value={formatMs(stats.latency_p50_ms)}
                    />
                    <Stat
                        label="Latency p95"
                        value={formatMs(stats.latency_p95_ms)}
                    />
                </div>
                <Stat label="Total cost" value={formatCost(stats.total_cost_usd)} />
            </CardContent>
        </Card>
    );
}

function Stat({ label, value }: { label: string; value: string }): React.JSX.Element {
    return (
        <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">{label}</span>
            <span className="text-lg font-semibold tabular-nums">{value}</span>
        </div>
    );
}

type FilterRowProps<T extends string> = {
    label: string;
    tabs: ReadonlyArray<{ value: T; label: string }>;
    active: T;
    onSelect: (next: T) => void;
};

function FilterRow<T extends string>({
    label,
    tabs,
    active,
    onSelect,
}: FilterRowProps<T>): React.JSX.Element {
    return (
        <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">{label}:</span>
            <div role="tablist" className="flex gap-1">
                {tabs.map((tab) => {
                    const isActive = tab.value === active;
                    return (
                        <button
                            key={tab.value}
                            type="button"
                            role="tab"
                            aria-selected={isActive}
                            onClick={() => { onSelect(tab.value); }}
                            className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
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
        </div>
    );
}

function CallList({
    query,
}: {
    query: ReturnType<typeof useLLMCalls>;
}): React.JSX.Element {
    if (query.isPending) {
        return <p className="text-muted-foreground">Loading calls...</p>;
    }
    if (query.isError) {
        return (
            <p className="text-destructive">
                Failed to load calls: {query.error.message}
            </p>
        );
    }
    const { rows, limit_reached } = query.data;
    if (rows.length === 0) {
        return (
            <Card>
                <CardContent className="p-6">
                    <p className="text-sm text-muted-foreground">
                        No calls match this filter.
                    </p>
                </CardContent>
            </Card>
        );
    }
    return (
        <div className="flex flex-col gap-3">
            <div className="overflow-x-auto rounded-md border border-border">
                <table className="w-full text-left text-sm">
                    <thead className="border-b border-border bg-muted/50 text-xs text-muted-foreground">
                        <tr>
                            <th className="p-2 font-medium">Transport</th>
                            <th className="p-2 font-medium">Method</th>
                            <th className="p-2 font-medium tabular-nums">Latency</th>
                            <th className="p-2 font-medium tabular-nums">Tokens</th>
                            <th className="p-2 font-medium">Status</th>
                            <th className="p-2 font-medium">When</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row) => (
                            <CallRow key={row.id} row={row} />
                        ))}
                    </tbody>
                </table>
            </div>
            {limit_reached ? (
                <p className="text-xs text-muted-foreground">
                    Showing the most recent calls. Older ones are not shown.
                </p>
            ) : null}
        </div>
    );
}

function CallRow({ row }: { row: LLMCallRow }): React.JSX.Element {
    const tokens =
        row.prompt_tokens === null && row.completion_tokens === null
            ? "—"
            : `${row.prompt_tokens ?? "—"}/${row.completion_tokens ?? "—"}`;
    return (
        <tr className="border-b border-border last:border-0">
            <td className="p-2">{row.transport_kind}</td>
            <td className="p-2 font-mono text-xs">{row.method}</td>
            <td className="p-2 tabular-nums">{formatMs(row.latency_ms)}</td>
            <td className="p-2 tabular-nums">{tokens}</td>
            <td className="p-2">
                {row.success ? (
                    <span className="text-success">ok</span>
                ) : (
                    <span className="text-destructive" title={row.error ?? undefined}>
                        error
                    </span>
                )}
            </td>
            <td className="p-2 text-xs text-muted-foreground">
                {formatWhen(row.created_at)}
            </td>
        </tr>
    );
}

function formatMs(ms: number | null): string {
    if (ms === null) {
        return "—";
    }
    if (ms >= 1000) {
        return `${(ms / 1000).toFixed(1)}s`;
    }
    return `${ms}ms`;
}

function formatRate(rate: number): string {
    return `${(rate * 100).toFixed(1)}%`;
}

function formatCost(cost: number): string {
    if (cost === 0) {
        return "—";
    }
    return `$${cost.toFixed(4)}`;
}

function formatWhen(iso: string): string {
    // ISO string straight from the API. Render as a local datetime
    // without pulling in a date library: the native toLocaleString
    // is enough for an ops table.
    const date = new Date(iso);
    return date.toLocaleString();
}
