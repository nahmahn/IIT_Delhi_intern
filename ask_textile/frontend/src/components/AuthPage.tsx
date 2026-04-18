import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";

type AuthMode = "login" | "register";

type Props = {
  onLogin: (
    token: string,
    user: { id: string; email: string; displayName?: string },
  ) => void;
};

const AuthPage = ({ onLogin }: Props) => {
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const endpoint =
        authMode === "login" ? "/api/v1/auth/login" : "/api/v1/auth/register";
      const body: any = { email, password };
      if (authMode === "register") body.displayName = displayName;

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.message || `Request failed with ${res.status}`);
      }

      const data = await res.json();
      const token = data.token;
      if (!token) {
        if (authMode === "register") {
          setError("Registration successful. Please login.");
          setAuthMode("login");
          return;
        }
        throw new Error("No token returned from server");
      }

      onLogin(token, data.user ?? { id: "", email, displayName });
      localStorage.setItem("authToken", token);
      navigate("/chat/new");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-950 p-4">
      <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900/90 p-7 shadow-xl backdrop-blur">
        <h1 className="text-3xl font-bold text-white mb-4 text-center">
          {authMode === "login" ? "Sign In" : "Create account"}
        </h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          {authMode === "register" && (
            <input
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="Display name"
              className="w-full rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-white focus:border-cyan-500 focus:outline-none"
            />
          )}
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            className="w-full rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-white focus:border-cyan-500 focus:outline-none"
            required
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            className="w-full rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-white focus:border-cyan-500 focus:outline-none"
            required
          />
          {error && <p className="text-sm text-red-400">{error}</p>}
          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-cyan-500 px-4 py-2 font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-50"
          >
            {loading
              ? "Please wait..."
              : authMode === "login"
                ? "Sign In"
                : "Register"}
          </button>
        </form>

        <p className="mt-4 text-center text-slate-300">
          {authMode === "login"
            ? "Don’t have an account?"
            : "Already have an account?"}{" "}
          <button
            className="font-semibold text-cyan-400 hover:text-cyan-300"
            onClick={() => {
              setError(null);
              setAuthMode(authMode === "login" ? "register" : "login");
            }}
          >
            {authMode === "login" ? "Register" : "Sign In"}
          </button>
        </p>
      </div>
    </div>
  );
};

export default AuthPage;
