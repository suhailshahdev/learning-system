import { useState } from "react";

import { CodeBlockView } from "@/components/session/code-block-view";
import { renderText } from "@/components/session/render-text";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { ParsedTurn } from "@/lib/api";

type Props = {
    turn: ParsedTurn;
    onSubmit: (answer: string) => void;
    isSubmitting: boolean;
    submitError: string | null;
};

/**
 * Renders a teaching turn and an answer form.
 *
 * Pure presentation plus local textarea state. The parent owns the
 * mutation and the response handling. After the split-roundtrip flow,
 * TurnView no longer needs an internal phase machine: grading lands
 * in its own component (GradingView) and the parent orchestrates
 * which to show.
 */
export function TurnView({
    turn,
    onSubmit,
    isSubmitting,
    submitError,
}: Props): React.JSX.Element {
    const [answer, setAnswer] = useState("");

    const handleSubmit = (event: React.SubmitEvent<HTMLFormElement>): void => {
        event.preventDefault();
        const trimmed = answer.trim();
        if (trimmed.length === 0) {
            return;
        }
        onSubmit(trimmed);
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
                {renderText(turn.question)}
            </div>
            {turn.question_code !== null ? (
                <CodeBlockView block={turn.question_code} />
            ) : null}
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
                    disabled={isSubmitting}
                    required
                />
                <Button
                    type="submit"
                    disabled={isSubmitting || answer.trim().length === 0}
                    className="self-start"
                >
                    {isSubmitting ? "Sending..." : "Send"}
                </Button>
                {submitError !== null ? (
                    <p className="text-sm text-destructive">
                        Failed to send: {submitError}
                    </p>
                ) : null}
            </form>
        </article>
    );
}
