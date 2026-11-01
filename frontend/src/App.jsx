import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8001";

function Icon({ children }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width="22"
      height="22"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

const TOPICS = [
  {
    title: "Internet problems",
    hint: "Slow, dropping, or no connection",
    prompt: "My internet is slow and keeps disconnecting",
    icon: (
      <Icon>
        <path d="M2 8.5a15 15 0 0 1 20 0M5 12a10 10 0 0 1 14 0M8.5 15.5a5 5 0 0 1 7 0" />
        <circle cx="12" cy="19" r="1" fill="currentColor" />
      </Icon>
    ),
  },
  {
    title: "Billing",
    hint: "Charges, due dates, payments",
    prompt: "I was charged more than I expected this month",
    icon: (
      <Icon>
        <path d="M6 3h12v18l-3-2-3 2-3-2-3 2z" />
        <path d="M9 8h6M9 12h6" />
      </Icon>
    ),
  },
  {
    title: "Router and Wi-Fi",
    hint: "Reset, reconnect, set up",
    prompt: "How do I reset my router and reconnect my devices?",
    icon: (
      <Icon>
        <rect x="3" y="13" width="18" height="7" rx="2" />
        <path d="M7 16.5h.01M11 16.5h.01M8 13V9M16 13V9" />
      </Icon>
    ),
  },
  {
    title: "Plans",
    hint: "Change, upgrade, or cancel",
    prompt: "I want to change or cancel my plan",
    icon: (
      <Icon>
        <path d="M4 8h14l-3-3M20 16H6l3 3" />
      </Icon>
    ),
  },
];

const TIPS = [
  "Say what you see: blinking lights, error codes, or messages.",
  "Mention which device is affected, or whether it's all of them.",
  "Tell me when the problem started.",
];

function SendIcon() {
  return (
    <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
      <path
        d="M10 16V4m0 0L5 9m5-5 5 5"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const isEmpty = messages.length === 0;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, error]);

  // Grow the textarea with its content, up to a limit
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [query]);

  async function sendMessage(customerQuery) {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: customerQuery, top_k: 5 }),
      });

      if (!response.ok) {
        throw new Error(`The server returned an error (${response.status}).`);
      }

      const data = await response.json();

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer,
          sources: Array.isArray(data.sources) ? data.sources : [],
        },
      ]);
    } catch (err) {
      setError({
        query: customerQuery,
        message:
          err instanceof TypeError
            ? "Can't reach the support service. Check that the backend is running."
            : err.message,
      });
    } finally {
      setLoading(false);
    }
  }

  async function ask(text) {
    const customerQuery = text.trim();
    if (!customerQuery || loading) return;

    setMessages((previous) => [
      ...previous,
      { role: "user", content: customerQuery },
    ]);
    setQuery("");

    await sendMessage(customerQuery);
  }

  function handleSubmit(event) {
    event.preventDefault();
    ask(query);
  }

  function startOver() {
    if (loading) return;
    setMessages([]);
    setQuery("");
    setError(null);
    textareaRef.current?.focus();
  }

  const remaining = 2000 - query.length;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            V
          </span>
          <span className="brand-name">Verizon Support</span>
        </div>

        <button type="button" className="new-chat" onClick={startOver}>
          <span aria-hidden="true">+</span> New conversation
        </button>

        <div className="tips">
          <h2>Get a faster answer</h2>
          <ul>
            {TIPS.map((tip) => (
              <li key={tip}>{tip}</li>
            ))}
          </ul>
        </div>

        <p className="sidebar-note">
          Answers are AI-generated. Check anything important, like charges or
          plan changes, with an agent.
        </p>
      </aside>

      <main className="chat">
        <header className="chat-header">
          <div className="chat-title">
            <span className="brand-mark small" aria-hidden="true">
              V
            </span>
            <div>
              <h1>Support assistant</h1>
              <p>
                <span className="status-dot" /> Online
              </p>
            </div>
          </div>
          <button
            type="button"
            className="header-reset"
            onClick={startOver}
            disabled={loading}
          >
            New chat
          </button>
        </header>

        <div className="messages" aria-live="polite">
          <div className={`messages-inner${isEmpty ? " empty" : ""}`}>
            {isEmpty && (
              <section className="welcome">
                <h2>How can we help today?</h2>
                <p>
                  Describe the problem in your own words, or start with one of
                  these.
                </p>
                <div className="topic-grid">
                  {TOPICS.map((topic) => (
                    <button
                      type="button"
                      className="topic"
                      key={topic.title}
                      onClick={() => ask(topic.prompt)}
                      disabled={loading}
                    >
                      <span className="topic-icon">{topic.icon}</span>
                      <span className="topic-text">
                        <strong>{topic.title}</strong>
                        <span>{topic.hint}</span>
                      </span>
                    </button>
                  ))}
                </div>
              </section>
            )}

            {messages.map((message, index) => (
              <div className={`row ${message.role}`} key={index}>
                {message.role === "assistant" && (
                  <span className="avatar" aria-hidden="true">
                    V
                  </span>
                )}

                <div className="bubble">
                  <span className="sr-only">
                    {message.role === "user" ? "You: " : "Assistant: "}
                  </span>

                  <p>{message.content}</p>

                  {message.role === "assistant" &&
                    message.sources?.length > 0 && (
                      <details className="sources">
                        <summary>
                          Retrieved support examples ({message.sources.length})
                        </summary>

                        <div className="sources-list">
                          {message.sources.map((source, sourceIndex) => (
                            <article
                              className="source-case"
                              key={`${source.rank}-${source.customer_message}`}
                            >
                              <div className="source-meta">
                                <strong>Source #{source.rank}</strong>
                                <span>
                                  RRF score:{" "}
                                  {Number(source.rrf_score).toFixed(4)}
                                </span>
                              </div>

                              <p>
                                <strong>Customer message</strong>
                                {source.customer_message}
                              </p>

                              <p>
                                <strong>Historical support response</strong>
                                {source.support_response}
                              </p>
                            </article>
                          ))}
                        </div>
                      </details>
                    )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="row assistant">
                <span className="avatar" aria-hidden="true">
                  V
                </span>
                <div className="bubble typing" role="status">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                  <span className="sr-only">Assistant is typing</span>
                </div>
              </div>
            )}

            {error && (
              <div className="error" role="alert">
                <div>
                  <strong>Message not sent</strong>
                  <p>{error.message}</p>
                </div>
                <button
                  type="button"
                  disabled={loading}
                  onClick={() => sendMessage(error.query)}
                >
                  Try again
                </button>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <div className="composer-box">
            <label className="sr-only" htmlFor="customer-query">
              Describe your issue
            </label>
            <textarea
              id="customer-query"
              ref={textareaRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Describe your internet, billing, or service issue"
              rows={1}
              maxLength={2000}
              disabled={loading}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form.requestSubmit();
                }
              }}
            />
            <button
              type="submit"
              className="send"
              disabled={loading || !query.trim()}
              aria-label="Send message"
            >
              <SendIcon />
            </button>
          </div>
          <div className="composer-hint">
            <span>Enter to send, Shift+Enter for a new line</span>
            {remaining < 200 && <span>{remaining} characters left</span>}
          </div>
        </form>
      </main>
    </div>
  );
}

export default App;
