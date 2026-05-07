import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { LearnedItemSummary } from "@/lib/api";

type DueForReviewProps = {
    items: LearnedItemSummary[];
};

export function DueForReview({ items }: DueForReviewProps): React.JSX.Element {
    return (
        <Card>
            <CardHeader>
                <CardTitle>Due for review</CardTitle>
                <CardDescription>
                    Items ordered by oldest review first.
                </CardDescription>
            </CardHeader>
            <CardContent>
                {items.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                        Nothing to review yet. Items appear here once you have learned topics.
                    </p>
                ) : (
                    <ul className="flex flex-col gap-3">
                        {items.map((item) => (
                            <li key={item.id} className="flex flex-col gap-1">
                                <p className="text-sm font-medium">{item.question}</p>
                                <p className="text-xs text-muted-foreground">
                                    {item.topic_path}
                                    {item.difficulty !== null ? ` · ${item.difficulty}` : ""}
                                    {` · ${item.mode}`}
                                </p>
                            </li>
                        ))}
                    </ul>
                )}
            </CardContent>
        </Card>
    );
}
