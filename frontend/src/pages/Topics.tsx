import { useMemo, useState } from "react";
import { Link } from "react-router";

import { ApiStatus } from "@/components/api-status";
import { ModeToggle } from "@/components/mode-toggle";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useTopics, type DomainSummary, type TopicSummary } from "@/lib/api";

const STATUS_LABELS: Record<TopicSummary["status"], string> = {
    not_started: "not started",
    in_progress: "in progress",
    learned: "learned",
    needs_revision: "needs revision",
};

const STATUS_STYLES: Record<TopicSummary["status"], string> = {
    not_started: "bg-muted text-muted-foreground",
    in_progress: "bg-success/15 text-success",
    learned: "bg-success/15 text-success",
    needs_revision: "bg-warning/15 text-warning",
};

export function Topics(): React.JSX.Element {
    const topics = useTopics();

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
                    <Link to="/search" className="underline underline-offset-4">
                        Search
                    </Link>
                </nav>
                <div className="flex items-center gap-4">
                    <ApiStatus />
                    <ModeToggle />
                </div>
            </header>
            <main className="p-8">
                {topics.isPending ? (
                    <p className="text-center text-muted-foreground">Loading...</p>
                ) : topics.isError ? (
                    <div className="mx-auto flex w-full max-w-md flex-col gap-2">
                        <Card>
                            <CardHeader>
                                <CardTitle>Could not load topics</CardTitle>
                                <CardDescription>{topics.error.message}</CardDescription>
                            </CardHeader>
                        </Card>
                    </div>
                ) : (
                    <TopicsTree
                        domains={topics.data.domains}
                        topics={topics.data.topics}
                    />
                )}
            </main>
        </div>
    );
}

type TopicsTreeProps = {
    domains: DomainSummary[];
    topics: TopicSummary[];
};

function TopicsTree({ domains, topics }: TopicsTreeProps): React.JSX.Element {
    const [expandedDomains, setExpandedDomains] = useState<Set<string>>(() => new Set());
    const [expandedTopics, setExpandedTopics] = useState<Set<string>>(() => new Set());

    // Group topics by domain name for fast lookup during render.
    const topicsByDomain = useMemo(() => {
        const map = new Map<string, TopicSummary[]>();
        for (const topic of topics) {
            const existing = map.get(topic.domain);
            if (existing === undefined) {
                map.set(topic.domain, [topic]);
            } else {
                existing.push(topic);
            }
        }
        return map;
    }, [topics]);

    // Build parent_id -> children map per domain for recursive
    // rendering. Topics with parent_id null are domain-level roots.
    // The empty domains case is handled by the outer render.
    const childrenByParent = useMemo(() => {
        const map = new Map<string, TopicSummary[]>();
        for (const topic of topics) {
            const key = topic.parent_id ?? `__root__:${topic.domain}`;
            const existing = map.get(key);
            if (existing === undefined) {
                map.set(key, [topic]);
            } else {
                existing.push(topic);
            }
        }
        return map;
    }, [topics]);

    const toggleDomain = (name: string): void => {
        setExpandedDomains((prev) => {
            const next = new Set(prev);
            if (next.has(name)) {
                next.delete(name);
            } else {
                next.add(name);
            }
            return next;
        });
    };

    const toggleTopic = (id: string): void => {
        setExpandedTopics((prev) => {
            const next = new Set(prev);
            if (next.has(id)) {
                next.delete(id);
            } else {
                next.add(id);
            }
            return next;
        });
    };

    if (domains.length === 0) {
        return (
            <div className="mx-auto flex w-full max-w-2xl flex-col gap-4">
                <h1 className="text-3xl font-bold">Topics</h1>
                <Card>
                    <CardContent>
                        <p className="text-sm text-muted-foreground">
                            No domains seeded yet. Run db seed to populate the reference list.
                        </p>
                    </CardContent>
                </Card>
            </div>
        );
    }

    return (
        <div className="mx-auto flex w-full max-w-2xl flex-col gap-4">
            <h1 className="text-3xl font-bold">Topics</h1>
            <Card>
                <CardHeader>
                    <CardTitle>All domains and topics</CardTitle>
                    <CardDescription>
                        Click a domain to expand its topics. Click a topic with children to nest deeper.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <ul className="flex flex-col gap-3">
                        {domains.map((domain) => {
                            const domainTopics = topicsByDomain.get(domain.name) ?? [];
                            const isExpanded = expandedDomains.has(domain.name);
                            return (
                                <li key={domain.name} className="flex flex-col gap-2">
                                    <button
                                        type="button"
                                        onClick={() => { toggleDomain(domain.name); }}
                                        className="flex items-center gap-2 text-left"
                                    >
                                        <span className="font-mono text-xs text-muted-foreground w-3">
                                            {isExpanded ? "▾" : "▸"}
                                        </span>
                                        <span className="text-sm font-medium">{domain.name}</span>
                                        <span className="text-xs text-muted-foreground">
                                            ({domain.kind}, {domainTopics.length})
                                        </span>
                                    </button>
                                    {isExpanded ? (
                                        <DomainTopics
                                            domain={domain}
                                            childrenByParent={childrenByParent}
                                            expandedTopics={expandedTopics}
                                            onToggleTopic={toggleTopic}
                                        />
                                    ) : null}
                                </li>
                            );
                        })}
                    </ul>
                </CardContent>
            </Card>
        </div>
    );
}

type DomainTopicsProps = {
    domain: DomainSummary;
    childrenByParent: Map<string, TopicSummary[]>;
    expandedTopics: Set<string>;
    onToggleTopic: (id: string) => void;
};

function DomainTopics({
    domain,
    childrenByParent,
    expandedTopics,
    onToggleTopic,
}: DomainTopicsProps): React.JSX.Element {
    const roots = childrenByParent.get(`__root__:${domain.name}`) ?? [];

    if (roots.length === 0) {
        return (
            <p className="pl-5 text-xs text-muted-foreground">
                No topics yet. Topics appear here as you learn.
            </p>
        );
    }

    return (
        <ul className="flex flex-col gap-1 pl-5">
            {roots.map((topic) => (
                <li key={topic.id}>
                    <TopicNode
                        topic={topic}
                        childrenByParent={childrenByParent}
                        expandedTopics={expandedTopics}
                        onToggleTopic={onToggleTopic}
                        depth={0}
                    />
                </li>
            ))}
        </ul>
    );
}

type TopicNodeProps = {
    topic: TopicSummary;
    childrenByParent: Map<string, TopicSummary[]>;
    expandedTopics: Set<string>;
    onToggleTopic: (id: string) => void;
    depth: number;
};

function TopicNode({
    topic,
    childrenByParent,
    expandedTopics,
    onToggleTopic,
    depth,
}: TopicNodeProps): React.JSX.Element {
    const children = childrenByParent.get(topic.id) ?? [];
    const hasChildren = children.length > 0;
    const isExpanded = expandedTopics.has(topic.id);
    const statusLabel = STATUS_LABELS[topic.status];
    const statusClass = STATUS_STYLES[topic.status];

    const indentStyle = { paddingLeft: `${String(depth * 16)}px` };

    return (
        <div className="flex flex-col gap-1">
            <div
                className="flex items-start gap-2 py-0.5"
                style={indentStyle}
            >
                {hasChildren ? (
                    <button
                        type="button"
                        onClick={() => { onToggleTopic(topic.id); }}
                        className="font-mono text-xs text-muted-foreground w-3 self-center"
                    >
                        {isExpanded ? "▾" : "▸"}
                    </button>
                ) : (
                    <span className="font-mono text-xs text-muted-foreground w-3" aria-hidden="true">
                        ·
                    </span>
                )}
                <div className="flex flex-1 items-start justify-between gap-3">
                    <div className="flex flex-col gap-0.5">
                        <p className="text-sm">{topic.name}</p>
                        <p className="text-xs text-muted-foreground font-mono">
                            {topic.path}
                        </p>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                        {topic.difficulty !== null ? (
                            <span className="text-xs text-muted-foreground">
                                {topic.difficulty}
                            </span>
                        ) : null}
                        <span className={`rounded-md px-2 py-0.5 text-xs font-medium ${statusClass}`}>
                            {statusLabel}
                        </span>
                    </div>
                </div>
            </div>
            {hasChildren && isExpanded ? (
                <ul className="flex flex-col gap-1">
                    {children.map((child) => (
                        <li key={child.id}>
                            <TopicNode
                                topic={child}
                                childrenByParent={childrenByParent}
                                expandedTopics={expandedTopics}
                                onToggleTopic={onToggleTopic}
                                depth={depth + 1}
                            />
                        </li>
                    ))}
                </ul>
            ) : null}
        </div>
    );
}
