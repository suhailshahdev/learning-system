import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { DomainFocus } from "@/lib/api";

type FocusByDomainProps = {
    domains: DomainFocus[];
};

export function FocusByDomain({ domains }: FocusByDomainProps): React.JSX.Element {
    return (
        <Card>
            <CardHeader>
                <CardTitle>Focus by domain</CardTitle>
                <CardDescription>
                    Topics you have started but not yet finished, grouped by domain.
                </CardDescription>
            </CardHeader>
            <CardContent>
                {domains.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                        No in-progress topics. Start a session to begin learning.
                    </p>
                ) : (
                    <div className="flex flex-col gap-4">
                        {domains.map((focus) => (
                            <div key={focus.domain} className="flex flex-col gap-2">
                                <p className="text-sm font-medium">{focus.domain}</p>
                                <ul className="flex flex-col gap-1 pl-4">
                                    {focus.in_progress_topics.map((topic) => (
                                        <li key={topic.id} className="text-sm text-muted-foreground">
                                            {topic.path}
                                            {topic.difficulty !== null ? ` · ${topic.difficulty}` : ""}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        ))}
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
