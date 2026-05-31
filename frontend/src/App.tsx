import { Route, Routes } from "react-router";

import { Browse } from "@/pages/Browse";
import { Home } from "@/pages/Home";
import { LLMCalls } from "@/pages/LLMCalls";
import { Search } from "@/pages/Search";
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
      <Route path="/search" element={<Search />} />
      <Route path="/topics" element={<Topics />} />
      <Route path="/admin/llm-calls" element={<LLMCalls />} />
    </Routes>
  );
}

export default App;
