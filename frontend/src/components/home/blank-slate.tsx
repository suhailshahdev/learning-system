import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StartForm } from "@/components/session/start-form";

export function BlankSlate(): React.JSX.Element {
    return (
        <div className="mx-auto flex w-full max-w-md flex-col items-center gap-6 p-8">
            <h1 className="text-3xl font-bold">Learning System</h1>
            <Card className="w-full">
                <CardHeader>
                    <CardTitle>Start learning</CardTitle>
                    <CardDescription>
                        Pick a topic and start a session. The teacher will adjust difficulty
                        as you go.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <StartForm />
                </CardContent>
            </Card>
        </div>
    );
}
