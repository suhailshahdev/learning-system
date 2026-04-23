import { useHealth } from "@/lib/api"

type Tone = "neutral" | "ok" | "warn" | "error";

type Display = {
    tone: Tone;
    label: string;
    title: string;
}

function resolveDisplay(result: ReturnType<typeof useHealth>): Display {
    if (result.isPending) {
        return { tone: "neutral", label: "checking", title: "Checking backend health..." };
    }
    if (result.isError) {
        return {
            tone: "error",
            label: "down",
            title: `Backend unreachable : ${result.error.message}`,
        };
    }
    if (result.data.status === "degraded") {
        return { tone: "warn", label: "degraded", title: "Backend reachable but reporting degraded state" };
    }
    return { tone: "ok", label: "ok", title: "Backend healthy" };
}

const TONE_CLASSES: Record<Tone, string> = {
    neutral: "bg-muted-foreground",
    ok: "bg-green-500",
    warn: "bg-amber-500",
    error: "bg-red-500",
};

export function ApiStatus(): React.JSX.Element {
    const result = useHealth();
    const { tone, label, title } = resolveDisplay(result);

    return(
        <div
            className="flex items-center gap-2 text-sm text-muted-foreground"
            title={title}
        >
            <span
                className={`inline-block size-2 rounded-full ${TONE_CLASSES[tone]}`}
                aria-hidden="true"
            />
            <span>API: {label}</span>
        </div>
    )
}
