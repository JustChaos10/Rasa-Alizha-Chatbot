function renderVisualPanel(rawData) {
  if (!rawData || typeof rawData !== "object") return;
  const imageUrl = rawData.image_url || rawData.thumbnail_url;
  if (!imageUrl) return;

  const title = escapeHtml(rawData.title || "");
  const alt = escapeHtml(rawData.alt || title);
  const summary = escapeHtml(rawData.summary || "");
  const attribution = escapeHtml(rawData.attribution || "");
  const sourceUrl = rawData.source_url ? escapeHtml(rawData.source_url) : "";

  const captionSections = [];
  if (summary) {
    captionSections.push('<p class="visual-panel__summary">' + summary + "</p>");
  }
  if (attribution || sourceUrl) {
    const sourceLabel = attribution || "Source";
    const sourceMarkup = sourceUrl
      ? '<a href="' + sourceUrl + '" target="_blank" rel="noopener noreferrer">' + sourceLabel + "</a>"
      : sourceLabel;
    if (sourceMarkup) {
      captionSections.push('<p class="visual-panel__source">' + sourceMarkup + "</p>");
    }
  }

  const panelParts = [
    '<div class="visual-panel">',
    '  <div class="visual-panel__image-wrapper">',
    '    <img src="' + imageUrl + '" alt="' + alt + '" class="visual-panel__image" loading="lazy">',
    "  </div>",
  ];
  if (title) {
    panelParts.push('  <h4 class="visual-panel__title">' + title + "</h4>");
  }
  if (captionSections.length) {
    panelParts.push(captionSections.join(""));
  }
  panelParts.push("</div>");

  const $panel = $(panelParts.join(""));
  $panel.appendTo(".chats").hide().fadeIn(600);
  $panel.find("img").on("error", function() {
    const container = $(this).closest(".visual-panel");
    if (container && container.length) {
      container.remove();
    }
  });
  scrollToBottomOfResults();
  persistChat();
}

