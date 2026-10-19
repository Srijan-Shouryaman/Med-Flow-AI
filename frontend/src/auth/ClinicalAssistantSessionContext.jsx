import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";

const ClinicalAssistantSessionContext = createContext(null);

export function ClinicalAssistantSessionProvider({ children }) {
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const conversationCounter = useRef(1);

  const createConversation = useCallback((patientId = null) => {
    const number = conversationCounter.current++;
    const now = Date.now();
    const conversation = {
      id: `conversation-${now.toString(36)}-${number}`,
      title: `Conversation ${number}`,
      patientId,
      createdAt: now,
      updatedAt: now,
      messages: [],
    };

    setConversations((items) => [...items, conversation]);
    setActiveConversationId(conversation.id);
    return conversation;
  }, []);

  const value = useMemo(() => ({
    conversations,
    setConversations,
    activeConversationId,
    setActiveConversationId,
    createConversation,
  }), [activeConversationId, conversations, createConversation]);

  return (
    <ClinicalAssistantSessionContext.Provider value={value}>
      {children}
    </ClinicalAssistantSessionContext.Provider>
  );
}

export function useClinicalAssistantSession() {
  const context = useContext(ClinicalAssistantSessionContext);
  if (!context) throw new Error("useClinicalAssistantSession must be used within its provider.");
  return context;
}
