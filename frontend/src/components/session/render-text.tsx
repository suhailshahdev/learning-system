import { CodeBlockView } from "@/components/session/code-block-view";

/**
 * Render text with three inline code mechanisms.
 *
 *   1. [LCODE language=X]...[/LCODE] with newlines in body -> CodeBlockView
 *   2. [LCODE language=X]...[/LCODE] without newlines -> inline <code>
 *   3. `x` -> inline <code>
 *
 * Plain text parts return as raw strings. React flattens the array
 * in JSX. Block-level [LCODE] segments break the line via natural
 * document flow because CodeBlockView is a block-level element.
 *
 * Used by both TurnView (question) and GradingView (explanation),
 * which is why the function lives in its own file rather than as
 * a private helper of either component.
 */

const LCODE_PATTERN = /\[LCODE language=([^\]]+)\]([\s\S]*?)\[\/LCODE\]/;
const SPLIT_PATTERN = /(\[LCODE language=[^\]]+\][\s\S]*?\[\/LCODE\]|`[^`]+`)/g;

export function renderText(text: string): React.ReactNode {
    const parts = text.split(SPLIT_PATTERN);
    return parts.map((part, i) => {
        const lcodeMatch = LCODE_PATTERN.exec(part);
        if (lcodeMatch) {
            const [, rawLanguage, rawBody] = lcodeMatch;
            if (rawLanguage === undefined || rawBody === undefined) {
                return part;
            }
            const language = rawLanguage.trim();
            const body = rawBody.replace(/^\n+|\n+$/g, "");
            if (body.includes("\n")) {
                return <CodeBlockView key={i} block={{ language, body }} />;
            }
            return (
                <code
                    key={i}
                    className="rounded bg-muted px-1 py-0.5 font-mono text-[0.9em]"
                >
                    {body}
                </code>
            );
        }
        if (part.startsWith("`") && part.endsWith("`") && part.length >= 2) {
            return (
                <code
                    key={i}
                    className="rounded bg-muted px-1 py-0.5 font-mono text-[0.9em]"
                >
                    {part.slice(1, -1)}
                </code>
            );
        }
        return part;
    });
}
