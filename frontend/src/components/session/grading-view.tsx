import { CodeBlockView } from "@/components/session/code-block-view";
import { renderText } from "@/components/session/render-text";
import { Button } from "@/components/ui/button";
import type { GradingVerdict, ParsedGrading } from "@/lib/api";

type Props = {
    grading: ParsedGrading;
    onContinue: () => void;
    isContinuing: boolean;
    continueError: string | null;
};

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

export function GradingView({
    grading,
    onContinue,
    isContinuing,
    continueError,
}: Props): React.JSX.Element {
    return (
        <article className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
                <span
                    className={`inline-flex w-fit items-center rounded-md px-2 py-0.5 text-xs font-medium uppercase tracking-wide ${VERDICT_STYLES[grading.verdict]}`}
                >
                    {VERDICT_LABELS[grading.verdict]}
                </span>
                <p className="text-base leading-relaxed whitespace-pre-wrap">
                    {renderText(grading.explanation)}
                </p>
            </div>

            {grading.explanation_code !== null ? (
                <CodeBlockView block={grading.explanation_code} />
            ) : null}

            <Button
                type="button"
                onClick={onContinue}
                disabled={isContinuing}
                className="self-start"
            >
                {isContinuing ? "Loading next question..." : "Continue"}
            </Button>

            {continueError !== null ? (
                <p className="text-sm text-destructive">
                    Failed to load next question: {continueError}
                </p>
            ) : null}
        </article>
    );
}
