import type { ParsedSessionEnd } from "@/lib/api";

type Props = {
    sessionEnd: ParsedSessionEnd;
};

/**
 * Renders the session-end proposal in read-only transcript mode.
 *
 * The session-end marker terminates the transcript. Distinct
 * styling so the user sees a clear "end of session" signal
 * without an interactive Approve button.
 */
export function TranscriptSessionEndView({ sessionEnd }: Props): React.JSX.Element {
    return (
        <article className="flex flex-col gap-2 rounded-md border border-border bg-muted/30 p-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Session end
            </p>
            <p className="text-base leading-relaxed whitespace-pre-wrap">
                {sessionEnd.summary}
            </p>
        </article>
    );
}
