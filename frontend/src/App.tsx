import { Route, Routes } from "react-router";

import { Home } from "@/pages/Home";

function App(): React.JSX.Element {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
    </Routes>
  );
}

export default App;
