const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const messages = document.getElementById("messages");
const messagesEnd = document.getElementById("messagesEnd");
const welcomeState = document.getElementById("welcomeState");
const sendBtn = chatForm.querySelector(".send-btn");
const chatList = document.getElementById("chatList");
const chatSearchInput = document.getElementById("chatSearchInput");
const newChatBtn = document.getElementById("newChatBtn");
const siteModal = document.getElementById("siteModal");
const siteModalTitle = document.getElementById("siteModalTitle");
const siteModalMessage = document.getElementById("siteModalMessage");
const siteModalConfirm = document.querySelector("[data-modal-confirm]");
const siteModalCancel = document.querySelector("[data-modal-cancel]");
const isAuthenticated = document.body.dataset.authenticated === "true";
const chatStorageKey = "sistema_xia.chatHistory";
const draftStorageKey = `${chatStorageKey}.draft`;
let activeChatId = null;
let currentStorageKey = isAuthenticated ? draftStorageKey : chatStorageKey;
let sidebarChats = [];
let currentSearchQuery = "";
let searchDebounceId = null;
let scrollQueued = false;
let modalResolver = null;
let modalMode = "info";

function getCsrfToken() {
  const match = document.cookie.match(/(?:^|; )csrftoken=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : "";
}

function closeSiteModal(result = false) {
  if (!siteModal) {
    return;
  }

  siteModal.classList.remove("is-open");
  siteModal.setAttribute("aria-hidden", "true");
  modalMode = "info";

  if (modalResolver) {
    const resolve = modalResolver;
    modalResolver = null;
    resolve(result);
  }
}

function openSiteModal({ title, message, confirmText = "Aceptar", cancelText = "Cancelar", kind = "info" }) {
  if (!siteModal || !siteModalTitle || !siteModalMessage || !siteModalConfirm || !siteModalCancel) {
    return Promise.resolve(kind !== "confirm");
  }

  siteModalTitle.textContent = title;
  siteModalMessage.textContent = message;
  siteModalConfirm.textContent = confirmText;
  siteModalCancel.textContent = cancelText;
  siteModalCancel.style.display = kind === "confirm" ? "inline-flex" : "none";
  siteModalConfirm.classList.toggle("site-modal-button-primary", true);
  siteModalConfirm.classList.toggle("site-modal-button-secondary", false);
  siteModal.classList.add("is-open");
  siteModal.setAttribute("aria-hidden", "false");
  modalMode = kind;

  return new Promise((resolve) => {
    modalResolver = resolve;
  });
}

function showSiteMessage(title, message) {
  return openSiteModal({ title, message, kind: "info", confirmText: "Entendido" });
}

function askSiteConfirm(title, message, confirmText = "Eliminar") {
  return openSiteModal({ title, message, kind: "confirm", confirmText, cancelText: "Cancelar" });
}

function getStorageKey() {
  if (!isAuthenticated) {
    return chatStorageKey;
  }

  if (activeChatId) {
    return `${chatStorageKey}.${activeChatId}`;
  }

  return draftStorageKey;
}

function getStoredMessagesForKey(storageKey) {
  try {
    const raw = sessionStorage.getItem(storageKey);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderMarkdown(source) {
  const text = `${source || ""}`.replace(/\r\n/g, "\n");
  if (!text.trim()) {
    return "";
  }

  const lines = text.split("\n");
  const blocks = [];
  let currentParagraph = [];
  let currentList = null;
  let inCodeBlock = false;
  let codeLines = [];
  let index = 0;

  const isTableSeparator = (line) => /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line);

  const parseTableRow = (line) => line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());

  const flushParagraph = () => {
    if (!currentParagraph.length) return;
    blocks.push(`<p>${formatInline(currentParagraph.join("\n"))}</p>`);
    currentParagraph = [];
  };

  const flushList = () => {
    if (!currentList) return;
    const items = currentList.items.map((item) => `<li>${formatInline(item)}</li>`).join("");
    blocks.push(`<${currentList.tag}>${items}</${currentList.tag}>`);
    currentList = null;
  };

  const flushCodeBlock = () => {
    if (!codeLines.length) return;
    blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    codeLines = [];
  };

  while (index < lines.length) {
    const line = lines[index];

    if (line.trim().startsWith("```")) {
      if (inCodeBlock) {
        inCodeBlock = false;
        flushCodeBlock();
      } else {
        flushParagraph();
        flushList();
        inCodeBlock = true;
      }
      index += 1;
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      index += 1;
      continue;
    }

    const nextLine = lines[index + 1] || "";
    if (line.includes("|") && isTableSeparator(nextLine)) {
      flushParagraph();
      flushList();

      const headers = parseTableRow(line);
      const rows = [];
      index += 2;

      while (index < lines.length) {
        const rowLine = lines[index];
        if (!rowLine.includes("|") || !rowLine.trim()) {
          break;
        }
        rows.push(parseTableRow(rowLine));
        index += 1;
      }

      const headHtml = headers.map((cell) => `<th>${formatInline(cell)}</th>`).join("");
      const bodyHtml = rows.map((row) => `<tr>${row.map((cell) => `<td>${formatInline(cell)}</td>`).join("")}</tr>`).join("");
      blocks.push(`<table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`);
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = headingMatch[1].length;
      blocks.push(`<h${level}>${formatInline(headingMatch[2].trim())}</h${level}>`);
      index += 1;
      continue;
    }

    const unorderedMatch = line.match(/^[-*+]\s+(.*)$/);
    const orderedMatch = line.match(/^\d+\.\s+(.*)$/);
    if (unorderedMatch || orderedMatch) {
      flushParagraph();
      if (!currentList || currentList.tag !== (orderedMatch ? "ol" : "ul")) {
        flushList();
        currentList = { tag: orderedMatch ? "ol" : "ul", items: [] };
      }
      currentList.items.push((unorderedMatch || orderedMatch)[1].trim());
      index += 1;
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      index += 1;
      continue;
    }

    flushList();
    currentParagraph.push(line.trim());
    index += 1;
  }

  flushParagraph();
  flushList();
  if (inCodeBlock) {
    flushCodeBlock();
  }

  return blocks.join("");
}

function formatInline(value) {
  const safe = escapeHtml(value);
  return safe
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.+?)__/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/_(.+?)_/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/\[(.+?)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\n/g, "<br>");
}

function getStoredMessages() {
  return getStoredMessagesForKey(currentStorageKey);
}

function storeMessages(history) {
  sessionStorage.setItem(currentStorageKey, JSON.stringify(history));
}

function setActiveChatId(chatId, { preserveCurrentMessages = false } = {}) {
  activeChatId = chatId ? Number(chatId) : null;
  currentStorageKey = getStorageKey();

  if (preserveCurrentMessages) {
    const history = getStoredMessagesForKey(draftStorageKey);
    sessionStorage.setItem(currentStorageKey, JSON.stringify(history));
    sessionStorage.removeItem(draftStorageKey);
  }
}

function renderChatList() {
  if (!chatList) {
    return;
  }

  if (!sidebarChats.length) {
    chatList.innerHTML = currentSearchQuery
      ? '<li class="history-empty">No hay coincidencias para tu búsqueda.</li>'
      : '<li class="history-empty">Aún no tienes chats guardados.</li>';
    return;
  }

  chatList.innerHTML = sidebarChats.map((chat) => {
    const isActive = Number(chat.id) === Number(activeChatId);
    return `
      <li class="history-item ${isActive ? "is-active" : ""}">
        <button type="button" class="history-item-button" data-chat-id="${chat.id}">
          <span class="history-item-icon" aria-hidden="true"><i class="fa-solid fa-message"></i></span>
          <span class="history-item-body">
            <span class="history-item-title">${escapeHtml(chat.title || `Chat ${chat.id}`)}</span>
          </span>
        </button>
        <button type="button" class="history-item-delete" data-delete-chat-id="${chat.id}" aria-label="Eliminar chat ${escapeHtml(chat.title || `Chat ${chat.id}`)}" title="Eliminar chat">
          <i class="fa-solid fa-trash-can" aria-hidden="true"></i>
        </button>
      </li>
    `;
  }).join("");
}

async function loadChatList({ autoSelect = true, query = "" } = {}) {
  if (!isAuthenticated || !chatList) {
    return;
  }

  currentSearchQuery = query.trim();

  try {
    const url = query ? `/api/chats/?q=${encodeURIComponent(query)}` : "/api/chats/";
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error("No se pudo cargar el historial.");
    }

    const payload = await response.json();
    sidebarChats = Array.isArray(payload.chats) ? payload.chats : [];
    renderChatList();

    if (autoSelect && sidebarChats.length && !activeChatId) {
      await loadChat(sidebarChats[0].id);
    }
  } catch (error) {
    chatList.innerHTML = `<div class="history-empty">${escapeHtml(error.message)}</div>`;
  }
}

async function loadChat(chatId) {
  if (!isAuthenticated) {
    return;
  }

  try {
    const response = await fetch(`/api/chats/${chatId}/`);
    if (!response.ok) {
      throw new Error("No se pudo abrir el chat seleccionado.");
    }

    const payload = await response.json();
    const messagesPayload = Array.isArray(payload.messages) ? payload.messages : [];
    setActiveChatId(chatId);
    storeMessages(messagesPayload.map((item) => ({ role: item.role, content: item.content })));
    renderHistory();
    renderChatList();
  } catch (error) {
    appendMessage("assistant", `Error: ${error.message}`);
  }
}

function startNewChat() {
  if (!isAuthenticated) {
    return;
  }

  activeChatId = null;
  currentStorageKey = draftStorageKey;
  sessionStorage.removeItem(draftStorageKey);
  messages.innerHTML = "";
  updateWelcomeState();
  renderChatList();
}

function resolveChatIdFromResponse(response) {
  const headerChatId = response.headers.get("X-Chat-Id");
  if (headerChatId) {
    return Number(headerChatId);
  }

  return activeChatId;
}

function getConversationForRequest() {
  return getStoredMessages()
    .filter((item) => item && (item.role === "user" || item.role === "assistant") && `${item.content || ""}`.trim())
    .slice(-20)
    .map((item) => ({ role: item.role, content: `${item.content || ""}` }));
}

function updateWelcomeState() {
  const hasMessages = messages.children.length > 0;
  welcomeState.style.display = hasMessages ? "none" : "grid";
}

function scrollToBottom(behavior = "auto") {
  messagesEnd.scrollIntoView({ block: "end", behavior });
}

function queueScrollToBottom(behavior = "auto") {
  if (scrollQueued) {
    return;
  }

  scrollQueued = true;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      scrollToBottom(behavior);
      scrollQueued = false;
    });
  });
}

function createAvatar(role) {
  const avatar = document.createElement("div");
  avatar.className = `chat-avatar chat-avatar-${role}`;
  const icon = document.createElement("i");
  icon.className = role === "user" ? "fa-solid fa-user" : "fa-solid fa-robot";
  icon.setAttribute("aria-hidden", "true");
  avatar.appendChild(icon);
  return avatar;
}

function createBubble(role, content = "", isLoading = false) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble chat-bubble-${role}`;

  const contentEl = document.createElement("div");
  contentEl.className = "chat-content";

  if (role === "assistant" && isLoading) {
    contentEl.innerHTML = '<span class="chat-loading"><i class="fa-solid fa-spinner"></i> Pensando...</span>';
  } else if (role === "assistant") {
    contentEl.innerHTML = renderMarkdown(content);
  } else {
    contentEl.innerHTML = `<p>${escapeHtml(content).replace(/\n/g, "<br>")}</p>`;
  }

  bubble.appendChild(contentEl);
  return { bubble, contentEl };
}

function createMessageElement(role, content = "", options = {}) {
  const message = document.createElement("article");
  message.className = `chat-message chat-message-${role}`;

  const avatar = createAvatar(role);
  const { bubble, contentEl } = createBubble(role, content, options.loading);

  if (role === "user") {
    message.appendChild(bubble);
    message.appendChild(avatar);
  } else {
    message.appendChild(avatar);
    message.appendChild(bubble);
  }

  return { message, contentEl };
}

function appendMessage(role, content = "", options = {}) {
  const { message, contentEl } = createMessageElement(role, content, options);
  messages.appendChild(message);
  queueScrollToBottom();
  updateWelcomeState();
  return { message, contentEl };
}

function renderHistory() {
  messages.innerHTML = "";
  const history = getStoredMessages();
  for (const item of history) {
    appendMessage(item.role, item.content);
  }
  updateWelcomeState();
  queueScrollToBottom("auto");
}

function saveHistoryEntry(role, content) {
  const history = getStoredMessages();
  history.push({ role, content });
  storeMessages(history);
}

function updateLastAssistantMessage(content) {
  const assistantMessages = messages.querySelectorAll(".chat-message-assistant .chat-content");
  const target = assistantMessages[assistantMessages.length - 1];
  if (!target) return;
  target.innerHTML = renderMarkdown(content);

  const history = getStoredMessages();
  for (let index = history.length - 1; index >= 0; index -= 1) {
    if (history[index].role === "assistant") {
      history[index].content = content;
      break;
    }
  }
  storeMessages(history);
  queueScrollToBottom();
}

function replaceLastAssistantLoading(content) {
  const assistantMessages = messages.querySelectorAll(".chat-message-assistant .chat-content");
  const target = assistantMessages[assistantMessages.length - 1];
  if (!target) return;
  target.innerHTML = renderMarkdown(content);
  queueScrollToBottom();
}

const chatObserver = new MutationObserver(() => {
  queueScrollToBottom();
});

chatObserver.observe(messages, {
  childList: true,
  subtree: true,
  characterData: true,
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = chatInput.value.trim();
  if (!value) return;

  const hadChatBeforeSubmit = Boolean(activeChatId);

  appendMessage("user", value);
  saveHistoryEntry("user", value);
  chatInput.value = "";
  chatInput.style.height = "46px";

  const assistantEntry = appendMessage("assistant", "", { loading: true });
  saveHistoryEntry("assistant", "");
  sendBtn.disabled = true;

  try {
    const response = await fetch("/api/chat/stream/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: value,
        messages: getConversationForRequest(),
        ...(activeChatId ? { chat_id: activeChatId } : {}),
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error("No se pudo iniciar el stream del chat.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let completeText = "";

    while (true) {
      const { value: chunk, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const content = line.trim();
        if (!content) continue;

        const payload = JSON.parse(content);
        if (payload.error) {
          throw new Error(payload.error);
        }

        if (payload.chunk) {
          completeText += payload.chunk;
          replaceLastAssistantLoading(completeText);
        }
      }
    }

    if (!completeText.trim()) {
      completeText = "Sin respuesta del modelo.";
    }

    updateLastAssistantMessage(completeText);
    if (isAuthenticated) {
      const resolvedChatId = resolveChatIdFromResponse(response);
      if (resolvedChatId && (!hadChatBeforeSubmit || Number(resolvedChatId) !== Number(activeChatId))) {
        setActiveChatId(resolvedChatId, { preserveCurrentMessages: !hadChatBeforeSubmit });
        await loadChatList({ autoSelect: false });
      }
    }
  } catch (error) {
    const message = `Error: ${error.message}`;
    updateLastAssistantMessage(message);
  } finally {
    sendBtn.disabled = false;
  }
});

chatInput.addEventListener("input", () => {
  chatInput.style.height = "46px";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 132)}px`;
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) {
    return;
  }

  event.preventDefault();
  if (!sendBtn.disabled) {
    chatForm.requestSubmit();
  }
});

if (newChatBtn) {
  newChatBtn.addEventListener("click", startNewChat);
}

if (chatList) {
  chatList.addEventListener("click", async (event) => {
    const deleteBtn = event.target.closest(".history-item-delete");
    if (deleteBtn) {
      event.preventDefault();
      event.stopPropagation();

      const chatId = deleteBtn.getAttribute("data-delete-chat-id");
      const chat = sidebarChats.find((item) => String(item.id) === String(chatId));

      const confirmed = await askSiteConfirm(
        "Eliminar chat",
        `¿Eliminar el chat "${chat?.title || `Chat ${chatId}`}"? Esta acción no se puede deshacer.`,
        "Eliminar"
      );

      if (confirmed) {
        try {
          const response = await fetch(`/api/chats/${chatId}/delete/`, {
            method: "POST",
            headers: { "X-CSRFToken": getCsrfToken() },
          });

          if (!response.ok) throw new Error("No se pudo eliminar el chat.");

          if (Number(activeChatId) === Number(chatId)) {
            startNewChat();
          }

          await loadChatList({ autoSelect: false, query: chatSearchInput?.value || "" });
        } catch (error) {
          await showSiteMessage("Error", error.message);
        }
      }
      return;
    }

    const target = event.target.closest(".history-item-button");
    if (!target) return;

    const chatId = target.getAttribute("data-chat-id");
    if (chatId) {
      loadChat(chatId);
    }
  });
}

if (chatSearchInput) {
  chatSearchInput.addEventListener("input", () => {
    const query = chatSearchInput.value || "";

    if (searchDebounceId) {
      clearTimeout(searchDebounceId);
    }

    searchDebounceId = setTimeout(() => {
      loadChatList({ autoSelect: false, query });
    }, 180);
  });
}

if (isAuthenticated) {
  startNewChat();
  loadChatList({ autoSelect: false });
} else {
  sessionStorage.removeItem(chatStorageKey);
  currentStorageKey = chatStorageKey;
  renderHistory();
}

if (siteModal) {
  siteModal.addEventListener("click", (event) => {
    const closeTarget = event.target.closest("[data-modal-close]");
    if (closeTarget) {
      closeSiteModal(false);
      return;
    }

    if (event.target.closest("[data-modal-cancel]")) {
      closeSiteModal(false);
      return;
    }

    if (event.target.closest("[data-modal-confirm]")) {
      closeSiteModal(true);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && siteModal.classList.contains("is-open")) {
      closeSiteModal(false);
    }
  });
}

const sidebar = document.getElementById("sidebar");
const pinSidebarBtn = document.getElementById("pinSidebar");

if (pinSidebarBtn && sidebar) {
  pinSidebarBtn.addEventListener("click", () => {
    const isPinned = sidebar.classList.toggle("is-pinned");
    pinSidebarBtn.classList.toggle("active", isPinned);
    localStorage.setItem("sidebarPinned", isPinned);
  });

  if (localStorage.getItem("sidebarPinned") === "true") {
    sidebar.classList.add("is-pinned");
    pinSidebarBtn.classList.add("active");
  }
}
