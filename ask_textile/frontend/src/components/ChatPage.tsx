import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

type Role = "user" | "assistant" | "system";
type Message = { id: string; role: Role; content: string };
type Conversation = { id: string; title: string; updatedAt: string };

type Props = {
  token: string;
  onLogout: () => void;
};

const MAX_SUMMARY_CHARS = 2048;

const ChatPage = ({ token, onLogout }: Props) => {
  const { conversationId } = useParams<{ conversationId: string }>();
  const navigate = useNavigate();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversationId, setSelectedConversationId] = useState<
    string | null
  >(conversationId || null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [oldSummary, setOldSummary] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const storageKey = (id: string) => `ask_textile_history_${id}`;
  const summaryKey = (id: string) => `ask_textile_summary_${id}`;

  const trimSummary = (value: string) =>
    value.length > MAX_SUMMARY_CHARS
      ? value.slice(value.length - MAX_SUMMARY_CHARS)
      : value;

  const getLocalHistory = (id: string): Message[] => {
    try {
      const raw = localStorage.getItem(storageKey(id));
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed;
    } catch {
      // ignore malformed local storage
    }
    return [];
  };

  const getLocalSummary = (id: string): string => {
    try {
      return localStorage.getItem(summaryKey(id)) || "";
    } catch {
      return "";
    }
  };

  const saveLocalHistory = (
    id: string,
    nextMessages: Message[],
    baseSummary: string,
  ) => {
    const latest = nextMessages.slice(-10);
    const overflow = nextMessages.slice(0, -10);
    const overflowText = overflow
      .map((m) => `${m.role}: ${m.content}`)
      .join("\n");
    const consolidated = trimSummary(
      [baseSummary, overflowText].filter(Boolean).join("\n"),
    );

    localStorage.setItem(storageKey(id), JSON.stringify(latest));
    localStorage.setItem(summaryKey(id), consolidated);
    setOldSummary(consolidated);
  };

  const getHeaders = () => ({
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  });

  const loadConversations = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/chat", { headers: getHeaders() });
      if (!res.ok)
        throw new Error(`Unable to fetch conversations (${res.status})`);
      const data = await res.json();
      setConversations(data);
    } catch (err) {
      setError((err as Error).message);
    }
  }, [token]);

  const loadMessages = useCallback(
    async (id: string) => {
      if (id.startsWith("new-")) {
        setMessages([]);
        setOldSummary("");
        return;
      }

      try {
        const res = await fetch(`/api/v1/chat/${id}/messages`, {
          headers: getHeaders(),
        });
        if (!res.ok)
          throw new Error(`Unable to fetch messages (${res.status})`);
        const body = await res.json();
        const serverMessages: Message[] = body.map((m: any) => ({
          id: m.id,
          role: m.role as Role,
          content: m.content,
        }));

        const local = getLocalHistory(id);
        const merged = [
          ...serverMessages,
          ...local.filter((m) => !serverMessages.some((s) => s.id === m.id)),
        ];

        const baseSummary = getLocalSummary(id);
        const toSummarize = merged.slice(0, -10);
        const summaryText = trimSummary(
          [baseSummary, ...toSummarize.map((m) => `${m.role}: ${m.content}`)]
            .filter(Boolean)
            .join("\n"),
        );

        const trimmed = merged.slice(-10);
        setMessages(trimmed);
        saveLocalHistory(id, trimmed, summaryText);
      } catch (err) {
        setError((err as Error).message);
      }
    },
    [token],
  );

  useEffect(() => {
    if (!token) return;
    loadConversations();
  }, [token, loadConversations]);

  useEffect(() => {
    if (!selectedConversationId) return;

    // For new temporary conversations, start fresh
    if (selectedConversationId.startsWith("new-")) {
      setMessages([]);
      setOldSummary("");
      setPrompt("");
      setError(null);
      return;
    }

    // For saved conversations, load from server and local storage
    setMessages(getLocalHistory(selectedConversationId));
    setOldSummary(getLocalSummary(selectedConversationId));
    setPrompt("");
    setError(null);
    void loadMessages(selectedConversationId);
  }, [selectedConversationId, loadMessages]);

  const buildHistoryPayload = (nextMessages: Message[], summary: string) => {
    const history = nextMessages
      .slice(-10)
      .map((m) => ({ role: m.role, content: m.content }));
    const oldSummaryNormalized = summary.trim();
    const snippet = nextMessages
      .slice(-10)
      .map((m) => `${m.role}: ${m.content}`)
      .join(" \n");

    const combined = [oldSummaryNormalized, snippet].filter(Boolean).join("\n");

    return {
      history,
      summary: combined ? `Aggregated summary:\n${combined}` : "No history yet",
    };
  };

  const sendMessage = async (messageText: string) => {
    if (!selectedConversationId) return;
    setLoading(true);
    setError(null);

    try {
      const { history, summary } = buildHistoryPayload(messages, oldSummary);
      const response = await fetch("/api/v1/chat", {
        method: "POST",
        headers: getHeaders(),
        body: JSON.stringify({
          prompt: messageText,
          conversationId: selectedConversationId,
          history,
          summary,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.message || `Send failed ${response.status}`);
      }

      const data = await response.json();
      const newMessages: Message[] = [
        ...messages,
        { id: `u-${Date.now()}`, role: "user", content: messageText },
        {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: data.answer ?? "No response",
        },
      ];

      const mergedSummary = trimSummary(
        [
          oldSummary,
          ...newMessages.slice(0, -10).map((m) => `${m.role}: ${m.content}`),
        ]
          .filter(Boolean)
          .join("\n"),
      );

      const trimmed = newMessages.slice(-10);
      setMessages(trimmed);
      setOldSummary(mergedSummary);
      saveLocalHistory(selectedConversationId, trimmed, mergedSummary);

      if (
        data.conversationId &&
        data.conversationId !== selectedConversationId
      ) {
        setSelectedConversationId(data.conversationId);
        navigate(`/chat/${data.conversationId}`);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

const createNewConversation = () => {
  const id = `new-${Date.now()}`;

  // ✅ Only clear the temp "new-" entries, not saved conversations
  Object.keys(localStorage).forEach((key) => {
    if (
      key.startsWith("ask_textile_history_new") ||
      key.startsWith("ask_textile_summary_new")
    ) {
      localStorage.removeItem(key);
    }
  });

  setSelectedConversationId(id);
  setSidebarOpen(false);
  navigate(`/chat/${id}`);
};

  const conversationItems = useMemo(() => conversations || [], [conversations]);
    console.log(conversationItems)
  return (
    <div className="min-h-screen grid grid-cols-1 lg:grid-cols-[280px_1fr] bg-slate-950 text-slate-100">
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 w-64 z-50 transform transition-transform duration-300 border-r border-slate-800 bg-slate-900 p-4 flex flex-col ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        } lg:static lg:translate-x-0 lg:w-auto lg:z-auto lg:transform-none`}
      >
        <div className="flex items-center gap-2 mb-4">
          <span className="text-lg">🧠</span>
          <h1 className="text-lg font-semibold">AskTextile</h1>
        </div>

        <div className="flex gap-2 mb-4">
          <button
            onClick={createNewConversation}
            className="rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-sm font-medium hover:bg-slate-700"
          >
            New conversation
          </button>
          <button
            onClick={() => {
              onLogout();
              localStorage.removeItem("authToken");
            }}
            className="rounded-md border border-rose-600 bg-rose-600/10 px-3 py-2 text-sm font-medium text-rose-200 hover:bg-rose-500/20"
          >
            Logout
          </button>
        </div>

        <div className="mt-2 flex-1 overflow-y-auto">
          <ul className="space-y-2">
            {conversationItems.map((conv) => (
              <li key={conv.id}>
                <button
                  onClick={() => {
                    setSelectedConversationId(conv.id);
                    navigate(`/chat/${conv.id}`);
                    setSidebarOpen(false);
                  }}
                  className={`w-full rounded-lg px-3 py-2 text-left text-sm font-medium transition ${
                    selectedConversationId === conv.id
                      ? "bg-cyan-500 text-white"
                      : "bg-slate-800 text-slate-200 hover:bg-slate-700"
                  }`}
                >
                  {conv.title || "Conversation"}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </aside>

      <main className="p-4 flex flex-col gap-4 bg-slate-950" role="main">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3 gap-2">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="lg:hidden p-2 hover:bg-slate-800 rounded-lg text-xl"
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? "✕" : "☰"}
          </button>
          <h2 className="text-lg font-semibold flex-1">
            {selectedConversationId?.startsWith("new-")
              ? "✨ New Chat"
              : `Chat: ${selectedConversationId || "Loading..."}`}
          </h2>
          <span className="text-sm text-slate-400">
            Summary len: {oldSummary.length}
          </span>
        </div>

        <div className="space-y-3 overflow-y-auto rounded-xl border border-slate-800 bg-slate-900 p-4 h-[75vh]">
          {messages.length === 0 ? (
            <div className="rounded-lg border border-slate-700 bg-slate-800 p-3 text-sm text-slate-300">
              No messages yet. Ask something to start.
            </div>
          ) : (
            messages.map((m) => (
              <div
                key={m.id}
                className={`rounded-xl p-3 border ${m.role === "user" ? "self-end rounded-br-none bg-cyan-500/20 border-cyan-500 text-cyan-100" : "self-start rounded-bl-none bg-slate-800 border-slate-700 text-slate-100"}`}
              >
                <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">
                  {m.role}
                </div>
                <div className="whitespace-pre-wrap">{m.content}</div>
              </div>
            ))
          )}
        </div>

        <div className="grid grid-cols-[1fr_auto] gap-2 h-[10vh]">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type a message..."
            className="w-full resize-none rounded-lg border border-slate-700 bg-slate-800 p-3 text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
            rows={3}
          />
          <button
            type="button"
            disabled={loading || !prompt.trim()}
            className="rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() => prompt.trim() && sendMessage(prompt.trim())}
          >
            {loading ? "Sending..." : "Send"}
          </button>
        </div>


      
      </main>
    </div>
  );
};

export default ChatPage;
