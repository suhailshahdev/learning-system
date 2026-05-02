import { ApiStatus } from "@/components/api-status";
import { ModeToggle } from "@/components/mode-toggle";
import { StartForm } from "@/components/session/start-form";

export function Home(): React.JSX.Element {
  return (
    <div className="min-h-svh bg-background text-foreground">
      <header className="flex items-center justify-end gap-4 p-4">
        <ApiStatus />
        <ModeToggle />
      </header>
      <main className="flex flex-col items-center gap-8 p-8">
        <h1 className="text-3xl font-bold">Learning System</h1>
        <StartForm />
      </main>
    </div>
  );
}
