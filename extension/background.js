// Background service worker: context menu + side panel wiring.
// Everything the user selects is handed to the side panel UI; no network
// calls happen here — the panel talks to the local backend directly.

const MENU_ID = "private-doc-ask";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: MENU_ID,
    title: 'Ask Private AI about "%s"',
    contexts: ["selection"],
  });
  // Let clicking the toolbar icon open the side panel.
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch(() => {});
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== MENU_ID || !info.selectionText) return;

  const payload = {
    text: info.selectionText,
    url: tab && tab.url,
    title: tab && tab.title,
    ts: Date.now(),
  };

  // Stash the selection so the panel can pick it up as soon as it loads
  // (opening the panel and messaging it race otherwise).
  await chrome.storage.session.set({ pendingSelection: payload });

  if (tab && tab.windowId != null) {
    try {
      await chrome.sidePanel.open({ windowId: tab.windowId });
    } catch (e) {
      // open() must be called in response to a user gesture; the context-menu
      // click qualifies, but guard just in case.
      console.warn("sidePanel.open failed", e);
    }
  }

  // Also try a live message for an already-open panel.
  chrome.runtime
    .sendMessage({ type: "selection", payload })
    .catch(() => {});
});
