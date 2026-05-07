import { Link } from "react-router";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { RecentSessionSummary } from "@/lib/api";

type ContinueLastProps = {
    session: RecentSessionSummary;
};

export function ContinueLast({ session }: ContinueLastProps): React.JSX.Element {
    const topicLabel = session.topic_path ?? "Unspecified topic";

    return (
        <Card>
            <CardHeader>
                <CardTitle>Continue your last session</CardTitle>
                <CardDescription>
                    {topicLabel}
                </CardDescription>
            </CardHeader>
            <CardContent className="flex items-center justify-between gap-4">
                <p className="text-xs text-muted-foreground">
                    {session.mode_used} · {session.transport_kind}
                </p>
                <Button asChild>
                    <Link to={`/session/${session.id}`}>Continue</Link>
                </Button>
            </CardContent>
        </Card>
    );
}
