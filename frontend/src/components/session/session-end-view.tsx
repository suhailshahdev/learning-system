import { useNavigate } from "react-router";

import { Button } from "@/components/ui/button";
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import {
    type ParsedSessionEnd,
    useAbandonSession,
    useApproveSession,
} from "@/lib/api";

type Props = {
    parsed: ParsedSessionEnd;
    sessionId: string;
};

export function SessionEndView({ parsed, sessionId }: Props): React.JSX.Element {
    const navigate = useNavigate();
    const approve = useApproveSession();
    const abandon = useAbandonSession();

    const isPending = approve.isPending || abandon.isPending;
    const lastError = abandon.error ?? approve.error;

    const handleApprove = (): void => {
        approve.mutate(
            { session_id: sessionId },
            {
                onSuccess: () => {
                    void navigate("/");
                },
            },
        );
    };

    const handleAbandon = (): void => {
        abandon.mutate(
            { session_id: sessionId },
            {
                onSuccess: () => {
                    void navigate("/");
                },
            },
        );
    };

    return (
        <Card>
            <CardHeader>
                <CardTitle>Session complete</CardTitle>
                <CardDescription>
                    The teacher proposed ending the session. Approve to mark
                    the items learned, or abandon to discard the session.
                </CardDescription>
            </CardHeader>
            <CardContent>
                <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {parsed.summary}
                </p>
            </CardContent>
            <CardFooter className="flex flex-col items-stretch gap-3">
                <div className="flex gap-2">
                    <Button
                        type="button"
                        onClick={handleApprove}
                        disabled={isPending}
                    >
                        {approve.isPending ? "Approving..." : "Approve"}
                    </Button>
                    <Button
                        type="button"
                        variant="outline"
                        onClick={handleAbandon}
                        disabled={isPending}
                    >
                        {abandon.isPending ? "Abandoning..." : "Abandon"}
                    </Button>
                </div>
                {lastError !== null ? (
                    <p className="text-sm text-destructive">
                        Failed: {lastError.message}
                    </p>
                ) : null}
            </CardFooter>
        </Card>
    );
}
