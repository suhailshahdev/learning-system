import { CodeBlockView } from "@/components/session/code-block-view";
import { renderText } from "@/components/session/render-text";
import type { ParsedTurn } from "@/lib/api";

type Props = {
    turn: ParsedTurn;
};

/**
 * Renders a teaching turn in read-only transcript mode.
 *
 * Same structure as TurnView from components/session/, minus the
 * answer form and submit button. The header is compact to fit the
 * vertical-stacked transcript flow.
 */
export function TranscriptTurnView({ turn }: Props): React.JSX.Element {
    return (
        <article className="flex flex-col gap-3 border-l-2 border-border pl-4">
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
        </article>
    );
}
