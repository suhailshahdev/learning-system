import { Route, Routes } from "react-router";

import { Home } from "@/pages/Home";
import { Session } from "@/pages/Session";
import { Topics } from "@/pages/Topics";

function App(): React.JSX.Element {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/session/:id" element={<Session />} />
      <Route path="/topics" element={<Topics />} />
    </Routes>
  );
}

export default App;
