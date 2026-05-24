import { useState } from "react";
import { Link } from "react-router";

import { ApiStatus } from "@/components/api-status";
import { ModeToggle } from "@/components/mode-toggle";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useIngestDocument, useSearch, type SearchHit } from "@/lib/api";

export function Search(): React.JSX.Element {
    const [query, setQuery] = useState("");
    const search = useSearch();

    const [docTitle, setDocTitle] = useState("");
    const [docContent, setDocContent] = useState("");
    const ingest = useIngestDocument();

    const onSubmit = (): void => {
        const trimmed = query.trim();
        if (trimmed.length === 0) {
            return;
        }
        search.mutate({ query: trimmed, limit: 10 });
    };

    const onIngest = (): void => {
        const title = docTitle.trim();
        const content = docContent.trim();
        if (title.length === 0 || content.length === 0) {
            return;
        }
        ingest.mutate(
            { title, content },
            {
                onSuccess: () => {
                    // Clear the form, ready for the next paste. The
                    // success message below reports what was ingested.
                    setDocTitle("");
                    setDocContent("");
                },
            },
        );
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
                <div className="mx-auto flex w-full max-w-2xl flex-col gap-4">
                    <h1 className="text-3xl font-bold">Search</h1>
                    <Card>
                        <CardHeader>
                            <CardTitle>Add a document</CardTitle>
                            <CardDescription>
                                Paste notes or an article to add it to the searchable corpus.
                                It is split into chunks and embedded.
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="flex flex-col gap-2">
                                <Input
                                    value={docTitle}
                                    onChange={(e) => { setDocTitle(e.target.value); }}
                                    placeholder="Title"
                                    aria-label="Document title"
                                />
                                <Textarea
                                    value={docContent}
                                    onChange={(e) => { setDocContent(e.target.value); }}
                                    placeholder="Paste the document text here..."
                                    aria-label="Document content"
                                    rows={6}
                                />
                                <div className="flex items-center gap-3">
                                    <Button
                                        onClick={onIngest}
                                        disabled={
                                            ingest.isPending
                                            || docTitle.trim().length === 0
                                            || docContent.trim().length === 0
                                        }
                                    >
                                        {ingest.isPending ? "Adding..." : "Add document"}
                                    </Button>
                                    {ingest.isSuccess ? (
                                        <p className="text-xs text-success">
                                            Added “{ingest.data.title}” — {ingest.data.chunk_count}{" "}
                                            {ingest.data.chunk_count === 1 ? "chunk" : "chunks"}.
                                        </p>
                                    ) : null}
                                    {ingest.isError ? (
                                        <p className="text-xs text-destructive">
                                            {ingest.error.message}
                                        </p>
                                    ) : null}
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardHeader>
                            <CardTitle>Semantic search</CardTitle>
                            <CardDescription>
                                Search your learned items and notes by meaning, not keywords.
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="flex gap-2">
                                <Input
                                    value={query}
                                    onChange={(e) => { setQuery(e.target.value); }}
                                    onKeyDown={(e) => {
                                        if (e.key === "Enter") {
                                            onSubmit();
                                        }
                                    }}
                                    placeholder="e.g. how do I add to a list"
                                    aria-label="Search query"
                                />
                                <Button
                                    onClick={onSubmit}
                                    disabled={search.isPending || query.trim().length === 0}
                                >
                                    {search.isPending ? "Searching..." : "Search"}
                                </Button>
                            </div>
                        </CardContent>
                    </Card>

                    {search.isError ? (
                        <Card>
                            <CardHeader>
                                <CardTitle>Search failed</CardTitle>
                                <CardDescription>{search.error.message}</CardDescription>
                            </CardHeader>
                        </Card>
                    ) : null}

                    {search.isSuccess ? (
                        <SearchResults hits={search.data.hits} />
                    ) : null}
                </div>
            </main>
        </div>
    );
}

type SearchResultsProps = {
    hits: SearchHit[];
};

function SearchResults({ hits }: SearchResultsProps): React.JSX.Element {
    if (hits.length === 0) {
        return (
            <Card>
                <CardContent>
                    <p className="text-sm text-muted-foreground">
                        No matches. The corpus may be empty, or nothing was similar enough.
                    </p>
                </CardContent>
            </Card>
        );
    }

    return (
        <ul className="flex flex-col gap-2">
            {hits.map((hit) => (
                <li key={hit.source_id}>
                    <Card>
                        <CardContent className="flex items-start justify-between gap-4 py-4">
                            <div className="flex flex-col gap-1">
                                <p className="text-sm whitespace-pre-wrap">{hit.content}</p>
                                <p className="text-xs text-muted-foreground">
                                    {hit.source_type === "learned_item" ? "learned item" : "document"}
                                </p>
                            </div>
                            <span className="shrink-0 rounded-md bg-muted px-2 py-0.5 text-xs font-mono text-muted-foreground">
                                {hit.score.toFixed(3)}
                            </span>
                        </CardContent>
                    </Card>
                </li>
            ))}
        </ul>
    );
}
