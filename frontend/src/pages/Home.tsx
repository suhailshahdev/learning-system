import { ApiStatus } from "@/components/api-status";
import { BlankSlate } from "@/components/home/blank-slate";
import { DueForReview } from "@/components/home/due-for-review";
import { FocusByDomain } from "@/components/home/focus-by-domain";
import { KnowledgeSummary } from "@/components/home/knowledge-summary";
import { ModeToggle } from "@/components/mode-toggle";
import { StartForm } from "@/components/session/start-form";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useHome } from "@/lib/api";

export function Home(): React.JSX.Element {
    const home = useHome();

    return (
        <div className="min-h-svh bg-background text-foreground">
            <header className="flex items-center justify-end gap-4 p-4">
                <ApiStatus />
                <ModeToggle />
            </header>
            <main className="p-8">
                {home.isPending ? (
                    <p className="text-center text-muted-foreground">Loading...</p>
                ) : home.isError ? (
                    <div className="mx-auto flex w-full max-w-md flex-col gap-2">
                        <Card>
                            <CardHeader>
                                <CardTitle>Could not load dashboard</CardTitle>
                                <CardDescription>{home.error.message}</CardDescription>
                            </CardHeader>
                        </Card>
                    </div>
                ) : home.data.is_blank_slate ? (
                    <BlankSlate />
                ) : (
                    <div className="mx-auto flex w-full max-w-2xl flex-col gap-6">
                        <h1 className="text-3xl font-bold">Learning System</h1>
                        <Card>
                            <CardHeader>
                                <CardTitle>Start a session</CardTitle>
                                <CardDescription>
                                    Pick a topic to begin or continue learning.
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <StartForm />
                            </CardContent>
                        </Card>
                        <DueForReview items={home.data.due_for_review} />
                        <FocusByDomain domains={home.data.focus_by_domain} />
                        <KnowledgeSummary rows={home.data.knowledge_summary} />
                    </div>
                )}
            </main>
        </div>
    );
}
