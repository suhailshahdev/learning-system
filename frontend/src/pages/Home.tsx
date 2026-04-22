import { ModeToggle } from "@/components/mode-toggle";

export function Home(): React.JSX.Element {
  return (
    <div className="min-h-svh bg-background text-foreground">
      <header className="flex justify-end p-4">
        <ModeToggle />
      </header>
      <main className="flex flex-col items-center justify-center gap-4 p-8">
        <h1 className="text-3xl font-bold">Learning System</h1>
        <p className="text-muted-foreground">
          Frontend skeleton. shadcn wired, dark mode working.
        </p>
      </main>
    </div>
  );
}
