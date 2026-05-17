import { Sparkles } from "lucide-react";
import { useState } from "react";
import { DiagnoseDialog } from "@/components/diagnose/diagnose-dialog";
import { Button } from "@/components/ui/button";
import { Dialog, DialogTrigger } from "@/components/ui/dialog";
import { useDiagnose } from "@/lib/api";

/**
 * Header entry point for the diagnostic flow.
 *
 * Opens the dialog in its idle (prompt) state. The user picks a
 * transport and triggers the diagnose call from inside the dialog.
 * This avoids spending tokens on accidental dialog opens.
 *
 * The transport the user picks here drives both the diagnose call
 * and the learning session that starts on accept.
 */
export function DiagnoseButton(): React.JSX.Element {
  const [open, setOpen] = useState(false);
  const diagnose = useDiagnose();

  const handleOpenChange = (next: boolean): void => {
    setOpen(next);
    if (!next) {
      diagnose.reset();
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Sparkles />
          What should I focus on?
        </Button>
      </DialogTrigger>
      <DiagnoseDialog
        diagnose={diagnose}
        onClose={() => { setOpen(false); }}
      />
    </Dialog>
  );
}
