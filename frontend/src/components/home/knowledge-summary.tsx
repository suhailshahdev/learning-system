import { useMemo } from "react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { KnowledgeSummaryRow } from "@/lib/api";

type KnowledgeSummaryProps = {
    rows: KnowledgeSummaryRow[];
};

type GroupedDomain = {
    domain: string;
    counts: { difficulty: string; count: number }[];
};

function groupByDomain(rows: KnowledgeSummaryRow[]): GroupedDomain[] {
    const grouped = new Map<string, { difficulty: string; count: number }[]>();
    for (const row of rows) {
        const existing = grouped.get(row.domain);
        if (existing === undefined) {
            grouped.set(row.domain, [{ difficulty: row.difficulty, count: row.count }]);
        } else {
            existing.push({ difficulty: row.difficulty, count: row.count });
        }
    }
    return Array.from(grouped.entries()).map(([domain, counts]) => ({ domain, counts }));
}

export function KnowledgeSummary({ rows }: KnowledgeSummaryProps): React.JSX.Element {
    const grouped = useMemo(() => groupByDomain(rows), [rows]);

    return (
        <Card>
            <CardHeader>
                <CardTitle>What I know</CardTitle>
                <CardDescription>
                    Summary of asserted knowledge by domain and difficulty.
                </CardDescription>
            </CardHeader>
            <CardContent>
                {grouped.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                        No knowledge assertions yet. Complete sessions to build this summary.
                    </p>
                ) : (
                    <ul className="flex flex-col gap-2">
                        {grouped.map(({ domain, counts }) => (
                            <li key={domain} className="flex flex-col gap-1">
                                <p className="text-sm font-medium">{domain}</p>
                                <p className="text-xs text-muted-foreground">
                                    {counts.map((c) => `${c.difficulty} (${String(c.count)})`).join(", ")}
                                </p>
                            </li>
                        ))}
                    </ul>
                )}
            </CardContent>
        </Card>
    );
}
