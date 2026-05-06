import type { CodeBlock } from "@/lib/api";

type Props = {
    block: CodeBlock;
};

/**
 * Renders a code block with a language label and monospace body.
 *
 * No syntax highlighting yet. The styling matches the requirement
 * block so code reads as supporting content rather than the main
 * focus.
 */
export function CodeBlockView({ block }: Props): React.JSX.Element {
    return (
        <div className="overflow-hidden rounded-md border border-border bg-muted/40">
            <div className="flex items-center border-b border-border px-3 py-1.5">
                <span className="text-xs uppercase tracking-wide text-muted-foreground">
                    {block.language}
                </span>
            </div>
            <pre className="overflow-x-auto p-3">
                <code className="font-mono text-sm leading-relaxed whitespace-pre">
                    {block.body}
                </code>
            </pre>
        </div>
    );
}
