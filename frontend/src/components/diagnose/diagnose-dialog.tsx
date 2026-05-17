import { AlertCircle, Info, Loader2 } from "lucide-react";
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
  ApiError,
  type DiagnoseRequest,
  type DiagnoseResponse,
  type TransportKind,
  useStartSession,
} from "@/lib/api";
import type { UseMutationResult } from "@tanstack/react-query";

type DiagnoseDialogProps = {
  diagnose: UseMutationResult<DiagnoseResponse, Error, DiagnoseRequest>;
  onClose: () => void;
};

/**
 * Dialog body for the diagnostic flow.
 *
 * Renders one of five views based on mutation state:
 *   - idle: transport selector and Diagnose button (initial)
 *   - pending: loading spinner
 *   - success: proposal with accept/reject (uses transport from idle view)
 *   - error (422): informational "no data yet" view
 *   - error (other): generic error
 *
 * The transport picked in the idle view drives both the diagnose
 * call and the learning session that starts on accept. State is
 * lifted into this component so all views see the same value.
 *
 * 422 is distinguished via ApiError.status because the backend
 * returns it for the empty-state case. It's not a system failure.
 * The user just hasn't built diagnosable history yet.
 */
export function DiagnoseDialog({
  diagnose,
  onClose,
}: DiagnoseDialogProps): React.JSX.Element {
  const [transportKind, setTransportKind] =
    useState<TransportKind>("deepseek");

  if (diagnose.isPending) {
    return <LoadingView />;
  }
  if (diagnose.isSuccess) {
    return (
      <ProposalView
        proposal={diagnose.data}
        transportKind={transportKind}
        onClose={onClose}
      />
    );
  }
  if (diagnose.isError) {
    const status =
      diagnose.error instanceof ApiError ? diagnose.error.status : undefined;
    if (status === 422) {
      const detail =
        diagnose.error instanceof ApiError
          ? diagnose.error.detail
          : undefined;
      return <NoDataView detail={detail} onClose={onClose} />;
    }
    return <ErrorView message={diagnose.error.message} onClose={onClose} />;
  }
  // Idle: show the prompt view with transport selector and trigger button.
  return (
    <PromptView
      transportKind={transportKind}
      onTransportChange={setTransportKind}
      onDiagnose={() => { diagnose.mutate({ transport_kind: transportKind }); }}
    />
  );
}

function LoadingView(): React.JSX.Element {
  return (
    <DialogContent>
      <DialogHeader>
        <DialogTitle>Looking at what you've been working on</DialogTitle>
        <DialogDescription>
          This takes a few seconds. The model is reading your recent
          sessions and weak topics.
        </DialogDescription>
      </DialogHeader>
      <div className="flex justify-center py-6">
        <Loader2 className="size-6 animate-spin text-muted-foreground" />
      </div>
    </DialogContent>
  );
}

type PromptViewProps = {
  transportKind: TransportKind;
  onTransportChange: (kind: TransportKind) => void;
  onDiagnose: () => void;
};

function PromptView({
  transportKind,
  onTransportChange,
  onDiagnose,
}: PromptViewProps): React.JSX.Element {
  return (
    <DialogContent>
      <DialogHeader>
        <DialogTitle>What should I focus on?</DialogTitle>
        <DialogDescription>
          The model will read your recent sessions, weak topics, and
          stale topics, then propose one focus area. The transport you
          pick here also runs the learning session if you accept.
        </DialogDescription>
      </DialogHeader>
      <div className="flex flex-col gap-2">
        <Label htmlFor="diagnose-transport">Transport</Label>
        <Select
          value={transportKind}
          onValueChange={(v) => { onTransportChange(v as TransportKind); }}
        >
          <SelectTrigger id="diagnose-transport" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="deepseek">DeepSeek</SelectItem>
            <SelectItem value="claude_playwright">Claude (Playwright)</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <DialogFooter>
        <Button onClick={onDiagnose}>Diagnose</Button>
      </DialogFooter>
    </DialogContent>
  );
}

type ProposalViewProps = {
  proposal: DiagnoseResponse;
  transportKind: TransportKind;
  onClose: () => void;
};

function ProposalView({
  proposal,
  transportKind,
  onClose,
}: ProposalViewProps): React.JSX.Element {
  const navigate = useNavigate();
  const startSession = useStartSession();

  const handleAccept = (): void => {
    startSession.mutate(
      { topic_path: proposal.topic_path, transport_kind: transportKind },
      {
        onSuccess: (data) => {
          onClose();
          void navigate(`/session/${data.session.id}`, {
            state: { firstTurn: data.first_turn, session: data.session },
          });
        },
      },
    );
  };

  return (
    <DialogContent>
      <DialogHeader>
        <DialogTitle>{proposal.topic_path}</DialogTitle>
        <DialogDescription>{proposal.reasoning}</DialogDescription>
      </DialogHeader>
      <p className="text-sm text-muted-foreground">
        Will start the learning session on{" "}
        <span className="font-medium text-foreground">
          {transportKind === "deepseek" ? "DeepSeek" : "Claude (Playwright)"}
        </span>
        .
      </p>
      {startSession.isError ? (
        <p className="text-sm text-destructive">
          Failed to start: {startSession.error.message}
        </p>
      ) : null}
      <DialogFooter>
        <Button
          variant="outline"
          onClick={onClose}
          disabled={startSession.isPending}
        >
          Not this one
        </Button>
        <Button onClick={handleAccept} disabled={startSession.isPending}>
          {startSession.isPending ? "Starting..." : "Start session"}
        </Button>
      </DialogFooter>
    </DialogContent>
  );
}

type NoDataViewProps = {
  detail: string | undefined;
  onClose: () => void;
};

function NoDataView({ detail, onClose }: NoDataViewProps): React.JSX.Element {
  return (
    <DialogContent>
      <DialogHeader>
        <DialogTitle className="flex items-center gap-2">
          <Info className="size-5 text-muted-foreground" />
          Nothing to diagnose yet
        </DialogTitle>
        <DialogDescription>
          {detail ?? "Start a learning session first to build diagnosable history."}
        </DialogDescription>
      </DialogHeader>
      <DialogFooter>
        <Button onClick={onClose}>OK</Button>
      </DialogFooter>
    </DialogContent>
  );
}

type ErrorViewProps = {
  message: string;
  onClose: () => void;
};

function ErrorView({ message, onClose }: ErrorViewProps): React.JSX.Element {
  return (
    <DialogContent>
      <DialogHeader>
        <DialogTitle className="flex items-center gap-2">
          <AlertCircle className="size-5 text-destructive" />
          Something went wrong
        </DialogTitle>
        <DialogDescription>{message}</DialogDescription>
      </DialogHeader>
      <DialogFooter>
        <Button onClick={onClose}>Close</Button>
      </DialogFooter>
    </DialogContent>
  );
}
