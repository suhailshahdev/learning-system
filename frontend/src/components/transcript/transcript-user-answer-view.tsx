type Props = {
    answer: string;
};

/**
 * Renders the user's historical answer in transcript mode.
 *
 * Plain text block with a subtle "Your answer" header. The
 * left-border styling distinguishes it from teaching turns:
 * teaching turns get the default border, user answers get a
 * darker accent so the eye can track question/answer pairs
 * down the page.
 */
export function TranscriptUserAnswerView({ answer }: Props): React.JSX.Element {
    return (
        <article className="flex flex-col gap-2 border-l-2 border-primary/40 pl-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Your answer
            </p>
            <p className="text-base leading-relaxed whitespace-pre-wrap">
                {answer}
            </p>
        </article>
    );
}
