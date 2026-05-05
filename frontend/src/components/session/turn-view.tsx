import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
    type GradingVerdict,
    type ParsedResponse,
    type ParsedTurn,
    useSendTurn,
} from "@/lib/api";

type Props = {
    turn: ParsedTurn;
    sessionId: string;
    onResponse: (parsed: ParsedResponse) => void;
};

type Phase = "feedback" | "answering";

const VERDICT_LABELS: Record<GradingVerdict, string> = {
    correct: "Correct",
    partial: "Partial",
    incorrect: "Incorrect",
    open_graded: "Open answer",
};

const VERDICT_STYLES: Record<GradingVerdict, string> = {
    correct: "bg-success/15 text-success",
    partial: "bg-warning/15 text-warning",
    incorrect: "bg-destructive/15 text-destructive",
    open_graded: "bg-muted text-muted-foreground",
};

export function TurnView({ turn, sessionId, onResponse }: Props): React.JSX.Element {
    // First turns have no grading since there is no previous answer.
    // Subsequent turns show feedback first and advance to the question
    // on continue.
    const hasGrading = turn.grading_verdict !== null;
    const [phase, setPhase] = useState<Phase>(hasGrading ? "feedback" : "answering");
    const [answer, setAnswer] = useState("");
    const sendTurn = useSendTurn();

    const handleContinue = (): void => {
        setPhase("answering");
    };

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

            {phase === "feedback" && turn.grading_verdict !== null ? (
                <FeedbackPanel
                    verdict={turn.grading_verdict}
                    explanation={turn.grading_explanation}
                    onContinue={handleContinue}
                />
            ) : (
                <AnsweringPanel
                    turn={turn}
                    answer={answer}
                    onAnswerChange={setAnswer}
                    onSubmit={handleSubmit}
                    isPending={sendTurn.isPending}
                    isError={sendTurn.isError}
                    errorMessage={sendTurn.error?.message ?? null}
                />
            )}
        </article>
    );
}

type FeedbackPanelProps = {
    verdict: GradingVerdict;
    explanation: string | null;
    onContinue: () => void;
};

function FeedbackPanel({
    verdict,
    explanation,
    onContinue,
}: FeedbackPanelProps): React.JSX.Element {
    return (
        <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
                <span
                    className={`inline-flex w-fit items-center rounded-md px-2 py-0.5 text-xs font-medium uppercase tracking-wide ${VERDICT_STYLES[verdict]}`}
                >
                    {VERDICT_LABELS[verdict]}
                </span>
                {explanation !== null ? (
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">
                        {explanation}
                    </p>
                ) : null}
            </div>

            <Button type="button" onClick={onContinue} className="self-start">
                Continue
            </Button>
        </div>
    );
}

type AnsweringPanelProps = {
    turn: ParsedTurn;
    answer: string;
    onAnswerChange: (value: string) => void;
    onSubmit: (event: React.SubmitEvent<HTMLFormElement>) => void;
    isPending: boolean;
    isError: boolean;
    errorMessage: string | null;
};

function AnsweringPanel({
    turn,
    answer,
    onAnswerChange,
    onSubmit,
    isPending,
    isError,
    errorMessage,
}: AnsweringPanelProps): React.JSX.Element {
    return (
        <>
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
            <form onSubmit={onSubmit} className="flex flex-col gap-3">
                <Label htmlFor="answer">Your answer</Label>
                <Textarea
                    id="answer"
                    value={answer}
                    onChange={(e) => { onAnswerChange(e.target.value); }}
                    placeholder="Type your answer..."
                    rows={4}
                    disabled={isPending}
                    required
                />
                <Button
                    type="submit"
                    disabled={isPending || answer.trim().length === 0}
                    className="self-start"
                >
                    {isPending ? "Sending..." : "Send"}
                </Button>
                {isError ? (
                    <p className="text-sm text-destructive">
                        Failed to send: {errorMessage}
                    </p>
                ) : null}
            </form>
        </>
    );
}
