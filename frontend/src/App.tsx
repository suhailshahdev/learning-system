import { Route, Routes } from "react-router";

import { Home } from "@/pages/Home";
import { Session } from "@/pages/Session";

function App(): React.JSX.Element {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/session/:id" element={<Session />} />
    </Routes>
  );
}

export default App;
