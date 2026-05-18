import { CodeBlockView } from "@/components/session/code-block-view";
import { renderText } from "@/components/session/render-text";
import type { GradingVerdict, ParsedGrading } from "@/lib/api";

type Props = {
    grading: ParsedGrading;
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

/**
 * Renders a grading response in read-only transcript mode.
 *
 * Same verdict-badge + explanation structure as GradingView from
 * components/session/, minus the Continue button.
 */
export function TranscriptGradingView({ grading }: Props): React.JSX.Element {
    return (
        <article className="flex flex-col gap-2 border-l-2 border-border pl-4">
            <span
                className={`inline-flex w-fit items-center rounded-md px-2 py-0.5 text-xs font-medium uppercase tracking-wide ${VERDICT_STYLES[grading.verdict]}`}
            >
                {VERDICT_LABELS[grading.verdict]}
            </span>
            <p className="text-base leading-relaxed whitespace-pre-wrap">
                {renderText(grading.explanation)}
            </p>
            {grading.explanation_code !== null ? (
                <CodeBlockView block={grading.explanation_code} />
            ) : null}
        </article>
    );
}
