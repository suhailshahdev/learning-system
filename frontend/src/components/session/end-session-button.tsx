import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { useAbandonSession } from "@/lib/api";

type Props = {
    sessionId: string;
    onAbandoned: () => void;
};

// How long the confirm state stays armed before reverting to the
// default state. If the user clicks once and walks away, a delayed
// second click should not abandon the session by accident. Five
// seconds is long enough to confirm on purpose but short enough
// that an old confirmation does not carry over.
const CONFIRM_TIMEOUT_MS = 5000;

export function EndSessionButton({ sessionId, onAbandoned }: Props): React.JSX.Element {
    const [confirming, setConfirming] = useState(false);
    const abandon = useAbandonSession();

    useEffect(() => {
        if (!confirming) {
            return;
        }
        const timer = window.setTimeout(() => {
            setConfirming(false);
        }, CONFIRM_TIMEOUT_MS);
        return () => {
            window.clearTimeout(timer);
        };
    }, [confirming]);

    const handleClick = (): void => {
        if (!confirming) {
            setConfirming(true);
            return;
        }
        abandon.mutate(
            { session_id: sessionId },
            {
                onSuccess: () => {
                    onAbandoned();
                },
            },
        );
    };

    const label = abandon.isPending
        ? "Ending..."
        : confirming
            ? "Click to confirm"
            : "End session";

    return (
        <div className="flex flex-col items-end gap-1">
            <Button
                type="button"
                variant={confirming ? "destructive" : "outline"}
                size="sm"
                onClick={handleClick}
                disabled={abandon.isPending}
            >
                {label}
            </Button>
            {abandon.isError ? (
                <p className="text-xs text-destructive">
                    Failed: {abandon.error.message}
                </p>
            ) : null}
        </div>
    );
}
