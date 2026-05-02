import { useState } from "react";
import { useNavigate } from "react-router";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import {
    type TransportKind,
    useStartSession,
} from "@/lib/api";

export function StartForm(): React.JSX.Element {
    const [topicPath, setTopicPath] = useState("");
    const [transportKind, setTransportKind] = useState<TransportKind>("deepseek");

    const navigate = useNavigate();
    const startSession = useStartSession();

    const handleSubmit = (event: React.SubmitEvent<HTMLFormElement>): void => {
        event.preventDefault();
        const trimmed = topicPath.trim();
        if (trimmed.length === 0) {
            return;
        }

        startSession.mutate(
            { topic_path: trimmed, transport_kind: transportKind },
            {
                onSuccess: (data) => {
                    void navigate(`/session/${data.session.id}`, {
                        state: { firstTurn: data.first_turn, session: data.session },
                    });
                },
            },
        );
    };

    return (
        <form onSubmit={handleSubmit} className="flex w-full max-w-md flex-col gap-4">
            <div className="flex flex-col gap-2">
                <Label htmlFor="topic-path">Topic</Label>
                <Input
                    id="topic-path"
                    type="text"
                    value={topicPath}
                    onChange={(e) => { setTopicPath(e.target.value); }}
                    placeholder="Python > Data Types > Integers"
                    required
                    minLength={1}
                    disabled={startSession.isPending}
                />
            </div>

            <div className="flex flex-col gap-2">
                <Label htmlFor="transport-kind">Transport</Label>
                <Select
                    value={transportKind}
                    onValueChange={(v) => { setTransportKind(v as TransportKind); }}
                    disabled={startSession.isPending}
                >
                    <SelectTrigger id="transport-kind" className="w-full">
                        <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="deepseek">DeepSeek</SelectItem>
                        <SelectItem value="claude_playwright">Claude (Playwright)</SelectItem>
                    </SelectContent>
                </Select>
            </div>

            <Button type="submit" disabled={startSession.isPending || topicPath.trim().length === 0}>
                {startSession.isPending ? "Starting..." : "Start session"}
            </Button>

            {startSession.isError ? (
                <p className="text-sm text-destructive">
                    Failed to start: {startSession.error.message}
                </p>
            ) : null}
        </form>
    );
}
