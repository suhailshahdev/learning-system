import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
    type ParsedResponse,
    type ParsedTurn,
    useSendTurn,
} from "@/lib/api";

type Props = {
    turn: ParsedTurn;
    sessionId: string;
    onResponse: (parsed: ParsedResponse) => void;
};

export function TurnView({ turn, sessionId, onResponse }: Props): React.JSX.Element {
    const [answer, setAnswer] = useState("");
    const sendTurn = useSendTurn();

    const handleSubmit = (event: React.SubmitEvent<HTMLFormElement>): void => {
        event.preventDefault();
        const trimmed = answer.trim();
        if (trimmed.length === 0) {
            return;
        }

        sendTurn.mutate(
            { session_id: sessionId, answer: trimmed },
            {
                onSuccess: (data) => {
                    onResponse(data.parsed);
                },
            },
        );
    };

    return (
        <article className="flex flex-col gap-4">
            <header className="flex flex-col gap-1">
                <p className="text-xs uppercase tracking-wide text-muted-foreground">
                    {turn.topic_path}
                </p>
                <p className="text-xs text-muted-foreground">
                    {turn.mode} · {turn.difficulty}
                </p>
            </header>

            <div className="text-base leading-relaxed whitespace-pre-wrap">
                {turn.question}
            </div>

            {turn.requirements !== null ? (
                <div className="rounded-md border border-border bg-muted/40 p-3 text-sm">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
                        Requirements
                    </p>
                    <p>{turn.requirements}</p>
                </div>
            ) : null}

            <form onSubmit={handleSubmit} className="flex flex-col gap-3">
                <Label htmlFor="answer">Your answer</Label>
                <Textarea
                    id="answer"
                    value={answer}
                    onChange={(e) => { setAnswer(e.target.value); }}
                    placeholder="Type your answer..."
                    rows={4}
                    disabled={sendTurn.isPending}
                    required
                />
                <Button
                    type="submit"
                    disabled={sendTurn.isPending || answer.trim().length === 0}
                    className="self-start"
                >
                    {sendTurn.isPending ? "Sending..." : "Send"}
                </Button>
                {sendTurn.isError ? (
                    <p className="text-sm text-destructive">
                        Failed to send: {sendTurn.error.message}
                    </p>
                ) : null}
            </form>
        </article>
    );
}
