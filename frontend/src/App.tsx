import { Route, Routes } from "react-router";

import { Browse } from "@/pages/Browse";
import { Home } from "@/pages/Home";
import { Session } from "@/pages/Session";
import { Topics } from "@/pages/Topics";
import { Transcript } from "@/pages/Transcript";

function App(): React.JSX.Element {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/session/:id" element={<Session />} />
      <Route path="/session/:id/transcript" element={<Transcript />} />
      <Route path="/sessions" element={<Browse />} />
      <Route path="/topics" element={<Topics />} />
    </Routes>
  );
}

export default App;
