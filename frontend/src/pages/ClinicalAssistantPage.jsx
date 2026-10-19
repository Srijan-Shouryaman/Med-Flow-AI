import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Bot,
  ChevronDown,
  LoaderCircle,
  Menu,
  Plus,
  RotateCw,
  Send,
  ShieldCheck,
  UserRound,
  X,
} from "lucide-react";
import { useLocation } from "react-router";
import { useAuth } from "../auth/AuthContext.jsx";
import { useClinicalAssistantSession } from "../auth/ClinicalAssistantSessionContext.jsx";
import { hasRoleAccess } from "../config/navigation.js";
import Card from "../components/ui/Card.jsx";
import FeedbackState from "../components/ui/FeedbackState.jsx";
import {
  clinicalAssistantApi,
  getClinicalQueryErrorMessage,
  parsePatientId,
} from "../services/clinicalAssistantApi.js";

const SUGGESTED_QUESTIONS = [
  "Summarize the documented clinical findings.",
  "What relevant history appears in the available records?",
  "List documented medications and investigations.",
];

const SOURCE_FIELDS = [
  ["patient_id", "Patient ID"],
  ["case_id", "Case ID"],
  ["report_id", "Report ID"],
  ["reranker_rank", "Reranker rank"],
];

function createId(prefix, counter) {
  const randomId = globalThis.crypto?.randomUUID?.();
  return randomId ? `${prefix}-${randomId}` : `${prefix}-${Date.now().toString(36)}-${counter}-${Math.random().toString(36).slice(2)}`;
}

function renderInlineText(text, keyPrefix) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return parts.map((part, index) => {
    const key = `${keyPrefix}-${index}`;
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={key}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`")) return <code key={key}>{part.slice(1, -1)}</code>;
    return <span key={key}>{part}</span>;
  });
}

function renderFormattedAnswer(answer) {
  const blocks = [];
  let paragraph = [];
  let listItems = [];
  let listKind = "";
  let itemKey = 0;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    blocks.push(<p key={`paragraph-${itemKey++}`}>{renderInlineText(paragraph.join(" "), `paragraph-inline-${itemKey}`)}</p>);
    paragraph = [];
  };

  const flushList = () => {
    if (!listItems.length) return;
    const List = listKind === "ordered" ? "ol" : "ul";
    blocks.push(
      <List key={`list-${itemKey++}`}>
        {listItems.map((item, index) => <li key={`list-item-${index}`}>{renderInlineText(item, `list-inline-${itemKey}-${index}`)}</li>)}
      </List>,
    );
    listItems = [];
    listKind = "";
  };

  for (const rawLine of answer.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = line.match(/^#{1,4}\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push(<h3 key={`heading-${itemKey++}`}>{renderInlineText(heading[1], `heading-inline-${itemKey}`)}</h3>);
      continue;
    }

    const bullet = line.match(/^[-*]\s+(.+)$/);
    const ordered = line.match(/^\d+[.)]\s+(.+)$/);
    const nextListKind = bullet ? "unordered" : ordered ? "ordered" : "";
    if (nextListKind) {
      flushParagraph();
      if (listKind && listKind !== nextListKind) flushList();
      listKind = nextListKind;
      listItems.push((bullet ?? ordered)[1]);
      continue;
    }

    flushList();
    paragraph.push(line);
  }

  flushParagraph();
  flushList();
  return blocks;
}

function formatConversationDate(value) {
  try {
    return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(value));
  } catch {
    return "Recent";
  }
}

function SourceList({ sources, citedChunkIds }) {
  const cited = new Set(citedChunkIds);
  const [isExpanded, setIsExpanded] = useState(false);
  const panelId = useId();
  return (
    <section className="clinical-source-section" aria-label="Clinical sources">
      <button
        className="clinical-source-heading"
        type="button"
        aria-expanded={isExpanded}
        aria-controls={panelId}
        aria-label={`${isExpanded ? "Collapse" : "Expand"} sources (${sources.length})`}
        onClick={() => setIsExpanded((expanded) => !expanded)}
      >
        <strong>Sources ({sources.length})</strong>
        <ChevronDown className={isExpanded ? "is-expanded" : ""} size={15} aria-hidden="true" />
      </button>
      <div
        id={panelId}
        className={`clinical-source-panel${isExpanded ? " is-expanded" : ""}`}
        aria-hidden={!isExpanded}
      >
        <div className="clinical-source-panel-inner">
          {sources.length ? (
            <div className="clinical-source-list">
              {sources.map((source, index) => {
                const chunkId = typeof source.chunk_id === "string" ? source.chunk_id : "";
                const isCited = chunkId && cited.has(chunkId);
                const visibleFields = SOURCE_FIELDS.filter(([key]) => source[key] !== null && source[key] !== undefined && source[key] !== "");
                return (
                  <article className="clinical-source-item" key={`${chunkId || "source"}-${index}`}>
                    <div className="clinical-source-item-header">
                      <span className="clinical-source-name">{chunkId || "Returned source metadata"}</span>
                      {isCited ? <span className="clinical-source-cited">Cited</span> : null}
                    </div>
                    {visibleFields.length ? (
                      <dl className="clinical-source-fields">
                        {visibleFields.map(([key, label]) => (
                          <div key={key}>
                            <dt>{label}</dt>
                            <dd>{String(source[key])}</dd>
                          </div>
                        ))}
                      </dl>
                    ) : <p className="clinical-source-unavailable">No source metadata was included.</p>}
                  </article>
                );
              })}
            </div>
          ) : <p className="clinical-source-empty">The response did not include source metadata.</p>}
        </div>
      </div>
    </section>
  );
}

function HumanReviewNotice({ status }) {
  const statusText = status === true
    ? "The backend requires qualified human review before clinical use."
    : status === false
      ? "The backend did not flag this response for human review; use clinical judgment before acting on generated content."
      : "The response did not include a review status. Treat this AI-generated content as requiring qualified human review.";

  return (
    <div className="clinical-review-notice" role="note">
      <ShieldCheck size={16} aria-hidden="true" />
      <span><strong>AI-generated summary.</strong> {statusText} This is not a confirmed diagnosis.</span>
    </div>
  );
}

function AssistantResponse({ result }) {
  return (
    <div className="clinical-assistant-response">
      <div className="clinical-answer-copy">{renderFormattedAnswer(result.answer)}</div>
      <HumanReviewNotice status={result.needsHumanReview} />
      <SourceList sources={result.sources} citedChunkIds={result.citedChunkIds} />
    </div>
  );
}

function EmptyChat({ onSuggest }) {
  return (
    <div className="clinical-chat-empty">
      <h2>How can I help review this patient's records?</h2>
      <p className="clinical-chat-empty-copy">Ask a question about the available clinical documentation. Answers are generated from retrieved records and include source details when available.</p>
      <div className="clinical-suggestion-list" aria-label="Suggested clinical questions">
        {SUGGESTED_QUESTIONS.map((question) => (
          <button key={question} type="button" onClick={() => onSuggest(question)}>{question}</button>
        ))}
      </div>
    </div>
  );
}

function ClinicalAssistantWorkspace() {
  const location = useLocation();
  const {
    conversations,
    setConversations,
    activeConversationId,
    setActiveConversationId,
    createConversation,
  } = useClinicalAssistantSession();
  const [patientIdInput, setPatientIdInput] = useState("");
  const [patientIdError, setPatientIdError] = useState("");
  const [composer, setComposer] = useState("");
  const [composerError, setComposerError] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const [pendingConversationId, setPendingConversationId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(() => typeof window === "undefined" || !window.matchMedia("(max-width: 900px)").matches);
  const messageCounter = useRef(1);
  const activeConversationIdRef = useRef(activeConversationId);
  const busyRef = useRef(false);
  const requestController = useRef(null);
  const inFlightRef = useRef(null);
  const mountedRef = useRef(false);
  const handledPatientNavigationRef = useRef(null);
  const bottomRef = useRef(null);
  const composerRef = useRef(null);

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeConversationId) ?? null,
    [activeConversationId, conversations],
  );
  const activeMessages = activeConversation?.messages ?? [];
  const activePatientId = activeConversation?.patientId ?? null;
  activeConversationIdRef.current = activeConversationId;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      const pending = inFlightRef.current;
      if (pending) {
        setConversations((items) => items.map((conversation) => conversation.id !== pending.conversationId ? conversation : {
          ...conversation,
          messages: conversation.messages.map((message) => message.id === pending.messageId
            ? { ...message, status: "failed", error: "The request was canceled when you left the assistant. Retry the question to continue." }
            : message),
        }));
      }
      requestController.current?.abort();
    };
  }, [setConversations]);

  useEffect(() => {
    setPatientIdInput(activeConversation?.patientId ? String(activeConversation.patientId) : "");
  }, [activeConversation?.patientId, activeConversationId]);

  useEffect(() => {
    const incomingPatientId = parsePatientId(location.state?.patientId);
    if (!incomingPatientId) return;

    const navigationKey = `${location.key}:${incomingPatientId}`;
    if (handledPatientNavigationRef.current === navigationKey) return;
    handledPatientNavigationRef.current = navigationKey;

    setPatientIdInput(String(incomingPatientId));
    setPatientIdError("");
    setComposerError("");

    const current = conversations.find((conversation) => conversation.id === activeConversationId);
    if (current?.patientId === incomingPatientId) return;

    if (current && current.patientId === null && current.messages.length === 0) {
      setConversations((items) => items.map((conversation) => conversation.id === current.id
        ? { ...conversation, patientId: incomingPatientId, updatedAt: Date.now() }
        : conversation));
      return;
    }

    createConversation(incomingPatientId);
  }, [location.key, location.state, conversations, activeConversationId, setConversations, createConversation]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [activeConversationId, activeMessages, pendingConversationId]);

  function updatePatientContext(event) {
    event.preventDefault();
    const patientId = parsePatientId(patientIdInput);
    if (!patientId) {
      setPatientIdError("Enter a positive whole-number Patient ID.");
      return;
    }

    setPatientIdError("");
    setComposerError("");
    setPatientIdInput(String(patientId));
    if (window.matchMedia("(max-width: 900px)").matches) setSidebarOpen(false);

    const current = conversations.find((conversation) => conversation.id === activeConversationId);
    if (current && current.patientId === patientId) return;

    if (current && current.patientId === null && current.messages.length === 0) {
      setConversations((items) => items.map((conversation) => conversation.id === current.id
        ? { ...conversation, patientId, updatedAt: Date.now() }
        : conversation));
      return;
    }

    createConversation(patientId);
  }

  function startNewChat() {
    createConversation(activePatientId);
    setPatientIdInput(activePatientId ? String(activePatientId) : "");
    setPatientIdError("");
    setComposerError("");
    if (window.matchMedia("(max-width: 900px)").matches) setSidebarOpen(false);
    composerRef.current?.focus();
  }

  function selectConversation(conversation) {
    setActiveConversationId(conversation.id);
    setConversations((items) => items.map((item) => item.id !== conversation.id ? item : {
      ...item,
      messages: item.messages.map((message) => message.receivedInBackground
        ? { ...message, receivedInBackground: false }
        : message),
    }));
    setPatientIdInput(conversation.patientId ? String(conversation.patientId) : "");
    setPatientIdError("");
    setComposerError("");
    if (window.matchMedia("(max-width: 900px)").matches) setSidebarOpen(false);
  }

  async function sendQuestion(questionValue, conversation = activeConversation, retryMessageId = null) {
    if (busyRef.current) return;
    if (!conversation || !conversation.patientId) {
      setPatientIdError("Set a valid Patient ID before asking a clinical question.");
      return;
    }

    const question = String(questionValue ?? "").trim();
    if (!question) {
      setComposerError("Enter a question before sending it.");
      return;
    }
    if (question.length > 5000) {
      setComposerError("Questions must be 5,000 characters or fewer.");
      return;
    }

    const messageId = retryMessageId ?? createId("message", messageCounter.current++);
    const controller = new AbortController();
    busyRef.current = true;
    requestController.current = controller;
    setIsBusy(true);
    setPendingConversationId(conversation.id);
    inFlightRef.current = { conversationId: conversation.id, messageId };
    setComposerError("");

    if (retryMessageId) {
      setConversations((items) => items.map((item) => item.id !== conversation.id ? item : {
        ...item,
        updatedAt: Date.now(),
        messages: item.messages.map((message) => message.id === retryMessageId
          ? { ...message, status: "pending", error: "" }
          : message),
      }));
    } else {
      const userMessage = { id: messageId, role: "doctor", content: question, status: "pending", error: "" };
      setConversations((items) => items.map((item) => item.id !== conversation.id ? item : {
        ...item,
        updatedAt: Date.now(),
        messages: [...item.messages, userMessage],
      }));
      setComposer("");
    }

    try {
      const result = await clinicalAssistantApi.query(conversation.patientId, question, { signal: controller.signal });
      if (!mountedRef.current) return;
      setConversations((items) => items.map((item) => item.id !== conversation.id ? item : {
        ...item,
        updatedAt: Date.now(),
        messages: [
          ...item.messages.map((message) => message.id === messageId ? { ...message, status: "sent", error: "" } : message),
          {
            id: createId("message", messageCounter.current++),
            role: "assistant",
            result,
            receivedInBackground: activeConversationIdRef.current !== conversation.id,
          },
        ],
      }));
    } catch (error) {
      if (!mountedRef.current || error?.name === "AbortError") return;
      const safeMessage = getClinicalQueryErrorMessage(error);
      setConversations((items) => items.map((item) => item.id !== conversation.id ? item : {
        ...item,
        updatedAt: Date.now(),
        messages: item.messages.map((message) => message.id === messageId
          ? { ...message, status: "failed", error: safeMessage }
          : message),
      }));
    } finally {
      if (mountedRef.current) {
        inFlightRef.current = null;
        requestController.current = null;
        busyRef.current = false;
        setIsBusy(false);
        setPendingConversationId(null);
      }
    }
  }

  function handleSend(event) {
    event.preventDefault();
    void sendQuestion(composer);
  }

  function handleComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void sendQuestion(composer);
    }
  }

  const sortedConversations = [...conversations].sort((first, second) => second.updatedAt - first.updatedAt);
  const activePending = pendingConversationId === activeConversationId;

  return (
    <div className={`clinical-chat-layout${sidebarOpen ? " has-sidebar" : " is-sidebar-collapsed"}`}>
      <aside className={`clinical-chat-sidebar${sidebarOpen ? " is-open" : ""}`} id="clinical-chat-sidebar" aria-label="Clinical assistant conversations">
        <div className="clinical-sidebar-heading">
          <strong>Patient context</strong>
          <button className="clinical-sidebar-close" type="button" aria-label="Hide conversations" onClick={() => setSidebarOpen(false)}><X size={17} aria-hidden="true" /></button>
        </div>

        <form className="clinical-patient-form" onSubmit={updatePatientContext} noValidate>
          <label htmlFor="clinical-patient-id">Patient ID</label>
          <div className="clinical-patient-input-row">
            <input
              id="clinical-patient-id"
              inputMode="numeric"
              autoComplete="off"
              value={patientIdInput}
              onChange={(event) => {
                setPatientIdInput(event.target.value);
                setPatientIdError("");
              }}
              aria-invalid={Boolean(patientIdError)}
              aria-describedby={patientIdError ? "clinical-patient-error" : "clinical-patient-help"}
              placeholder="Enter numeric patient ID"
            />
            <button className="clinical-set-patient-button" type="submit">Set</button>
          </div>
          <p id={patientIdError ? "clinical-patient-error" : "clinical-patient-help"} className={patientIdError ? "clinical-inline-error" : "clinical-patient-help"} role={patientIdError ? "alert" : undefined}>
            {patientIdError || (activePatientId ? `Selected patient · #${activePatientId}` : "A patient must be selected before querying records.")}
          </p>
        </form>

        <button className="clinical-new-chat-button" type="button" onClick={startNewChat}>
          <Plus size={16} aria-hidden="true" /> New Chat
        </button>

        <div className="clinical-history-heading">
          <span>Recent conversations</span>
          <span>{sortedConversations.length}</span>
        </div>
        <nav className="clinical-conversation-list" aria-label="Conversation history for this session">
          {sortedConversations.length ? sortedConversations.map((conversation) => (
            <button
              className={`clinical-conversation-button${conversation.id === activeConversationId ? " is-active" : ""}`}
              key={conversation.id}
              type="button"
              aria-current={conversation.id === activeConversationId ? "page" : undefined}
              onClick={() => selectConversation(conversation)}
            >
              <span className="clinical-conversation-icon"><UserRound size={14} aria-hidden="true" /></span>
              <span className="clinical-conversation-copy">
                <strong>{conversation.title}</strong>
                <small>{conversation.patientId ? `Patient #${conversation.patientId}` : "No patient selected"}</small>
              </span>
              <time dateTime={new Date(conversation.updatedAt).toISOString()}>{formatConversationDate(conversation.updatedAt)}</time>
            </button>
          )) : <p className="clinical-history-empty">Your conversations will appear here during this browser session.</p>}
        </nav>
      </aside>

      <section className="clinical-chat-main" aria-label="Clinical AI Assistant chat">
        <header className="clinical-chat-header">
          <button
            className="clinical-sidebar-toggle"
            type="button"
            aria-controls="clinical-chat-sidebar"
            aria-expanded={sidebarOpen}
            aria-label={sidebarOpen ? "Hide conversations" : "Show conversations"}
            onClick={() => setSidebarOpen((open) => !open)}
          >
            {sidebarOpen ? <X size={17} aria-hidden="true" /> : <Menu size={18} aria-hidden="true" />}
          </button>
          <div className="clinical-chat-title">
            <h1>Clinical AI Assistant</h1>
            <p>Evidence-grounded summaries from available patient records</p>
          </div>
          <div className={`clinical-patient-context${activePatientId ? " is-selected" : ""}`}>
            <span className="clinical-context-avatar"><UserRound size={15} aria-hidden="true" /></span>
            <span><small>Selected patient</small><strong>{activePatientId ? `#${activePatientId}` : "None"}</strong></span>
          </div>
        </header>

        <div className="clinical-chat-scroll" aria-live="polite">
          {activeMessages.length === 0 ? (
            <EmptyChat onSuggest={(question) => {
              setComposer(question);
              composerRef.current?.focus();
            }} />
          ) : (
            <div className="clinical-message-thread">
              {activeMessages.map((message) => (
                <article className={`clinical-message is-${message.role}`} key={message.id}>
                  <span className={`clinical-message-avatar is-${message.role}`}>
                    {message.role === "assistant" ? <Bot size={17} aria-hidden="true" /> : <UserRound size={16} aria-hidden="true" />}
                  </span>
                  <div className="clinical-message-body">
                    <div className="clinical-message-label">{message.role === "assistant" ? "Clinical AI Assistant" : "You"}</div>
                    {message.role === "doctor" ? (
                      <div className="clinical-doctor-bubble"><p>{message.content}</p></div>
                    ) : (
                      <>
                        {message.receivedInBackground ? <p className="clinical-stale-response-note" role="status">This response arrived while another conversation was open and was saved here.</p> : null}
                        <AssistantResponse result={message.result} />
                      </>
                    )}
                    {message.role === "doctor" && message.status === "failed" ? (
                      <div className="clinical-retry-row" role="alert">
                        <p>{message.error}</p>
                        <button type="button" disabled={isBusy} onClick={() => void sendQuestion(message.content, activeConversation, message.id)}>
                          <RotateCw size={13} aria-hidden="true" /> Retry
                        </button>
                      </div>
                    ) : null}
                  </div>
                </article>
              ))}
              {activePending ? (
                <article className="clinical-message is-assistant" aria-label="Generating response">
                  <span className="clinical-message-avatar is-assistant"><Bot size={17} aria-hidden="true" /></span>
                  <div className="clinical-message-body">
                    <div className="clinical-message-label">Clinical AI Assistant</div>
                    <div className="clinical-generating"><LoaderCircle size={16} className="is-spinning" aria-hidden="true" /><span>Reviewing retrieved records…</span></div>
                  </div>
                </article>
              ) : null}
              <div ref={bottomRef} />
            </div>
          )}
          {activeMessages.length === 0 ? <div ref={bottomRef} /> : null}
        </div>

        <div className="clinical-composer-wrap">
          {composerError ? <p className="clinical-composer-error" role="alert">{composerError}</p> : null}
          {!activePatientId ? (
            <p className="clinical-composer-prompt"><AlertTriangle size={14} aria-hidden="true" /> Set a valid Patient ID in the sidebar before sending a clinical question.</p>
          ) : null}
          <form className="clinical-composer" onSubmit={handleSend}>
            <textarea
              ref={composerRef}
              rows={2}
              maxLength={5000}
              value={composer}
              onChange={(event) => {
                setComposer(event.target.value);
                setComposerError("");
              }}
              onKeyDown={handleComposerKeyDown}
              disabled={!activePatientId || isBusy}
              aria-label="Your clinical question"
              placeholder={activePatientId ? "Ask a question about the available records…" : "Select a patient to begin…"}
            />
            <div className="clinical-composer-actions">
              <span>Enter to send · Shift + Enter for a new line</span>
              <button className="patient-primary-button clinical-send-button" type="submit" aria-label="Send question" disabled={!activePatientId || isBusy || !composer.trim()}>
                {isBusy ? <LoaderCircle size={15} className="is-spinning" aria-hidden="true" /> : <Send size={15} aria-hidden="true" />}
                <span>{isBusy ? "Sending" : "Send"}</span>
              </button>
            </div>
          </form>
        </div>
      </section>
    </div>
  );
}

export default function ClinicalAssistantPage() {
  const { user } = useAuth();
  const canAccessAssistant = hasRoleAccess(user, ["DOCTOR", "DEPARTMENT_HEAD", "HOSPITAL_ADMIN", "SUPER_ADMIN"]);

  if (!canAccessAssistant) {
    return (
      <Card className="clinical-assistant-access-card">
        <FeedbackState type="unauthorized" title="Clinical Assistant is unavailable for this account." description="Your role does not have access to clinical record queries." />
      </Card>
    );
  }

  return <ClinicalAssistantWorkspace />;
}
