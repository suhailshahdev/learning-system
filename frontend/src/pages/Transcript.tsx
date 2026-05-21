import { useState } from "react";
import { Link, useParams } from "react-router";

import { RetestDialog } from "@/components/retest/retest-dialog";
import { TranscriptGradingView } from "@/components/transcript/transcript-grading-view";
import { TranscriptSessionEndView } from "@/components/transcript/transcript-session-end-view";
import { TranscriptTurnView } from "@/components/transcript/transcript-turn-view";
import { TranscriptUserAnswerView } from "@/components/transcript/transcript-user-answer-view";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { useTranscript, type TranscriptEntry } from "@/lib/api/transcript";

export function Transcript(): React.JSX.Element {
    // useParams returns string | undefined regardless of generic.
    // Guard at use site.
    const { id } = useParams<{ id: string }>();
    const query = useTranscript(id);

    if (id === undefined) {
        return <NotLoaded message="No session id in the URL." />;
    }

    if (query.isPending) {
        return (
            <div className="min-h-svh bg-background text-foreground p-8">
                <p className="text-muted-foreground">Loading transcript...</p>
            </div>
        );
    }

    if (query.isError) {
        return <NotLoaded message={query.error.message} />;
    }

    const { session, entries } = query.data;

    return (
        <div className="min-h-svh bg-background text-foreground">
            <header className="flex items-center justify-between gap-4 border-b border-border p-4">
                <div className="flex flex-col gap-1">
                    <p className="text-sm text-muted-foreground">
                        Transcript · {session.state}
                    </p>
                    <p className="text-xs text-muted-foreground">
                        Session {id}
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    {session.state === "completed" ? (
                        <TranscriptRetestButton
                            sourceSessionId={id}
                            topicLabel="this session"
                        />
                    ) : null}
                    <Link
                        to="/"
                        className="text-sm underline underline-offset-4 text-muted-foreground hover:text-foreground"
                    >
                        Back to home
                    </Link>
                </div>
            </header>
            <main className="mx-auto flex w-full max-w-2xl flex-col gap-6 p-8">
                {entries.length === 0 ? (
                    <p className="text-muted-foreground">
                        This session has no recorded turns.
                    </p>
                ) : (
                    entries.map(renderEntry)
                )}
            </main>
        </div>
    );
}

function renderEntry(entry: TranscriptEntry): React.JSX.Element {
    // Pattern-matching on the kind discriminator. Each branch
    // narrows to its specific entry shape. Adding a new entry
    // kind on the backend surfaces here as a missing case at
    // compile time.
    switch (entry.kind) {
        case "turn":
            return <TranscriptTurnView key={entry.turn_index} turn={entry.turn} />;
        case "user_answer":
            return (
                <TranscriptUserAnswerView
                    key={entry.turn_index}
                    answer={entry.answer}
                />
            );
        case "grading":
            return (
                <TranscriptGradingView
                    key={entry.turn_index}
                    grading={entry.grading}
                />
            );
        case "session_end":
            return (
                <TranscriptSessionEndView
                    key={entry.turn_index}
                    sessionEnd={entry.session_end}
                />
            );
    }
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

type TranscriptRetestButtonProps = {
    sourceSessionId: string;
    topicLabel: string;
};

function TranscriptRetestButton({
    sourceSessionId,
    topicLabel,
}: TranscriptRetestButtonProps): React.JSX.Element {
    // Same per-button open state pattern as the Browse row button.
    // Conditional render (open ? <RetestDialog/> : null) makes the
    // dialog's internal mutation state reset between opens — close,
    // reopen, fresh prompt view.
    //
    // Eligibility is checked at the call site by verifying
    // state === "completed". The other condition of at least one
    // learned item is not visible from TranscriptResponse, so a
    // zero-item completed session would show the button and get
    // a 409 from the backend with empty_source. This is a rare
    // edge case and is handled by the dialog's error view.
    const [open, setOpen] = useState(false);

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <Button
                variant="outline"
                size="sm"
                onClick={() => { setOpen(true); }}
            >
                Retest this session
            </Button>
            {open ? (
                <RetestDialog
                    sourceSessionId={sourceSessionId}
                    topicLabel={topicLabel}
                    onClose={() => { setOpen(false); }}
                />
            ) : null}
        </Dialog>
    );
}
