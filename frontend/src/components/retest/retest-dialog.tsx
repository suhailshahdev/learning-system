import { AlertCircle, Loader2 } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router";

import { Button } from "@/components/ui/button";
import {
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import {
    useStartRetest,
    type StartRetestResponse,
    type TransportKind,
} from "@/lib/api";
import type { UseMutationResult } from "@tanstack/react-query";

type RetestDialogProps = {
    sourceSessionId: string;
    topicLabel: string;
    onClose: () => void;
};

/**
 * Dialog body for the retest flow.
 *
 * Owns the startRetest mutation directly (unlike DiagnoseDialog,
 * which receives the mutation lifted by its parent for the
 * two-step diagnose-then-start flow). Retest is one step: the
 * source session is already chosen by which button the user
 * clicked, so the dialog only needs the transport pick and the
 * mutation lives here.
 *
 * Renders one of three views based on mutation state:
 *   - idle: transport selector and Retest button
 *   - pending: loading spinner
 *   - error: generic error with retry
 *
 * Success navigates away (to the new retest session) and closes
 * the dialog, so there's no success view.
 *
 * Eligibility (source must be COMPLETED with >=1 item) is
 * checked at the call site before the button shows. The dialog
 * trusts the caller. Backend 409 still surfaces here as an
 * error if eligibility changed between button-click and
 * server-side check. The user can resolve this by closing
 * and not retrying.
 */
export function RetestDialog({
    sourceSessionId,
    topicLabel,
    onClose,
}: RetestDialogProps): React.JSX.Element {
    const startRetest = useStartRetest();
    const [transportKind, setTransportKind] = useState<TransportKind>("deepseek");

    if (startRetest.isPending) {
        return <LoadingView />;
    }
    if (startRetest.isError) {
        return (
            <ErrorView
                message={startRetest.error.message}
                onRetry={() => { startRetest.reset(); }}
                onClose={onClose}
            />
        );
    }
    return (
        <PromptView
            topicLabel={topicLabel}
            transportKind={transportKind}
            onTransportChange={setTransportKind}
            startRetest={startRetest}
            sourceSessionId={sourceSessionId}
            onClose={onClose}
        />
    );
}

type PromptViewProps = {
    topicLabel: string;
    transportKind: TransportKind;
    onTransportChange: (kind: TransportKind) => void;
    startRetest: UseMutationResult<
        StartRetestResponse,
        Error,
        { source_session_id: string; transport_kind: TransportKind }
    >;
    sourceSessionId: string;
    onClose: () => void;
};

function PromptView({
    topicLabel,
    transportKind,
    onTransportChange,
    startRetest,
    sourceSessionId,
    onClose,
}: PromptViewProps): React.JSX.Element {
    const navigate = useNavigate();

    const handleRetest = (): void => {
        startRetest.mutate(
            {
                source_session_id: sourceSessionId,
                transport_kind: transportKind,
            },
            {
                onSuccess: (data) => {
                    onClose();
                    void navigate(`/session/${data.session.id}`, {
                        state: {
                            firstTurn: data.first_turn,
                            session: data.session,
                        },
                    });
                },
            },
        );
    };

    return (
        <DialogContent>
            <DialogHeader>
                <DialogTitle>Retest this session</DialogTitle>
                <DialogDescription>
                    Walk through the same questions from{" "}
                    <span className="font-medium text-foreground">
                        {topicLabel}
                    </span>{" "}
                    again. Each answer is graded fresh by the model. The
                    original session stays untouched.
                </DialogDescription>
            </DialogHeader>
            <div className="flex flex-col gap-2">
                <Label htmlFor="retest-transport">Transport</Label>
                <Select
                    value={transportKind}
                    onValueChange={(v) => { onTransportChange(v as TransportKind); }}
                >
                    <SelectTrigger id="retest-transport" className="w-full">
                        <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="deepseek">DeepSeek</SelectItem>
                        <SelectItem value="claude_playwright">
                            Claude (Playwright)
                        </SelectItem>
                    </SelectContent>
                </Select>
            </div>
            <DialogFooter>
                <Button variant="outline" onClick={onClose}>
                    Cancel
                </Button>
                <Button onClick={handleRetest}>Start retest</Button>
            </DialogFooter>
        </DialogContent>
    );
}

function LoadingView(): React.JSX.Element {
    return (
        <DialogContent>
            <DialogHeader>
                <DialogTitle>Setting up the retest</DialogTitle>
                <DialogDescription>
                    Reconstructing the first question from your previous answers.
                </DialogDescription>
            </DialogHeader>
            <div className="flex justify-center py-6">
                <Loader2 className="size-6 animate-spin text-muted-foreground" />
            </div>
        </DialogContent>
    );
}

type ErrorViewProps = {
    message: string;
    onRetry: () => void;
    onClose: () => void;
};

function ErrorView({
    message,
    onRetry,
    onClose,
}: ErrorViewProps): React.JSX.Element {
    return (
        <DialogContent>
            <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                    <AlertCircle className="size-5 text-destructive" />
                    Couldn't start the retest
                </DialogTitle>
                <DialogDescription>{message}</DialogDescription>
            </DialogHeader>
            <DialogFooter>
                <Button variant="outline" onClick={onClose}>
                    Close
                </Button>
                <Button onClick={onRetry}>Try again</Button>
            </DialogFooter>
        </DialogContent>
    );
}
