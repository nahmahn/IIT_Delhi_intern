import { useEffect, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AuthPage from "./components/AuthPage";
import ChatPage from "./components/ChatPage";

function App() {
  const [token, setToken] = useState<string | null>(() =>
    localStorage.getItem("authToken"),
  );

  useEffect(() => {
    const savedToken = localStorage.getItem("authToken");
    if (savedToken && !token) setToken(savedToken);
  }, [token]);

  const handleLogin = (newToken: string) => {
    localStorage.setItem("authToken", newToken);
    setToken(newToken);
  };

  const handleLogout = () => {
    localStorage.removeItem("authToken");
    setToken(null);
  };

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/auth"
          element={
            token ? (
              <Navigate to="/chat/new" replace />
            ) : (
              <AuthPage onLogin={handleLogin} />
            )
          }
        />
        <Route
          path="/chat/:conversationId"
          element={
            token ? (
              <ChatPage token={token} onLogout={handleLogout} />
            ) : (
              <Navigate to="/auth" replace />
            )
          }
        />
        <Route
          path="/chat"
          element={<Navigate to={token ? "/chat/new" : "/auth"} replace />}
        />
        <Route
          path="/"
          element={<Navigate to={token ? "/chat/new" : "/auth"} replace />}
        />
        <Route
          path="*"
          element={<Navigate to={token ? "/chat/new" : "/auth"} replace />}
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
