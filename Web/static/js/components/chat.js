/**
 * scroll to the bottom of the chats after new message has been added to chat
 */
const converter = new showdown.Converter();
// Persist chat across navigations
function persistChat() {
  try {
    if (typeof sessionStorage !== 'undefined') {
      const chats = document.querySelector('.chats');
      if (chats) sessionStorage.setItem('chat_html', chats.innerHTML);
    }
  } catch (e) {}
}
function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
function hasRealChatMessages(chatsEl) {
  if (!chatsEl) return false;
  return !!chatsEl.querySelector(
    '.userMsg, .botMsg, .adaptive-card-wrapper, .imgcard, .pdf_attachment, .video-container, .suggestions, .quickReplies'
  );
}
function containsArabicChars(text) {
  return /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/.test(text || '');
}

function detectTextDirection(text) {
  return containsArabicChars(text) ? 'rtl' : 'ltr';
}

function applyInputDirectionFromValue() {
  try {
    const input = document.querySelector('.usrInput');
    if (!input) return;
    const direction = detectTextDirection(input.value);
    if (direction === 'rtl') {
      input.classList.add('rtl-text');
      input.setAttribute('dir', 'rtl');
    } else {
      input.classList.remove('rtl-text');
      input.setAttribute('dir', 'ltr');
    }
  } catch (e) {}
}

async function restoreChatFromStorage() {
  try {
    if (typeof sessionStorage === 'undefined') return;
    // Fetch current user
    let me = null;
    try {
      const resp = await fetch('/auth/me');
      me = await resp.json();
    } catch (e) { return; }
    if (!me || !me.authenticated || (me.role !== 'admin' && me.role !== 'Admin')) return;
    const email = me.email || '';
    const chatKey = `adminChat:${email}`;
    const flagKey = `adminReturnFlag:${email}`;
    if (!sessionStorage.getItem(flagKey)) return; // only on explicit return from admin
    const saved = sessionStorage.getItem(chatKey);
    const chats = document.querySelector('.chats');
    if (saved && chats && !hasRealChatMessages(chats) && !chats.dataset.restored) {
      chats.innerHTML = saved;
      chats.dataset.restored = 'true';
      ensureAvatarModalExists();
      bindPlayButtons();
      bindAvatarButtons();
      scrollToBottomOfResults();
    }
    // consume the flag so fresh reload doesn't always restore
    sessionStorage.removeItem(flagKey);
  } catch (e) {}
}

function getDefaultGreetingText() {
  const chatContainer = document.getElementById('chats');
  return chatContainer ? chatContainer.getAttribute('data-default-greeting') : null;
}

async function autoGreetIfNeeded() {
  try {
    if (typeof sessionStorage === 'undefined') return;
  } catch (e) {
    return;
  }

  const alreadyShown = sessionStorage.getItem('autoGreetingShown') === 'true';
  const storedChatHtml = sessionStorage.getItem('chat_html');
  const chatsEl = document.querySelector('.chats');
  const greetingText = getDefaultGreetingText();

  if (alreadyShown || (storedChatHtml && storedChatHtml.trim().length) || !greetingText || !chatsEl) {
    return;
  }

  if (hasRealChatMessages(chatsEl)) {
    return;
  }

  sessionStorage.setItem('autoGreetingShown', 'true');
  showBotTyping();
  setBotResponse([{ text: greetingText, metadata: { direction: 'ltr', source: 'auto_greeting' } }]);
  setTimeout(() => {
    try {
      persistChat();
    } catch (e) {}
  }, 800);
}

function scrollToBottomOfResults() {
  const terminalResultsDiv = document.getElementById("chats");
  terminalResultsDiv.scrollTop = terminalResultsDiv.scrollHeight;
}

function renderRelatedLinks(data) {
  if (!data || !Array.isArray(data.items) || data.items.length === 0) {
    return;
  }

  const safeItems = data.items
    .map((item) => {
      const title = escapeHtml(item.title || "");
      const prompt = item.prompt ? item.prompt.trim() : "";
      if (!title || !prompt) return null;
      return { title, prompt: prompt.slice(0, 240) };
    })
    .filter(Boolean);

  if (!safeItems.length) return;

  $(".related-links").remove();

  const lines = [];
  lines.push('<div class="related-links">');
  lines.push('  <div class="related-links__header" role="heading" aria-level="3">');
  lines.push('    <span class="material-icons related-links__icon" aria-hidden="true">list_alt</span>');
  lines.push('    <span class="related-links__title">Related</span>');
  lines.push("  </div>");
  lines.push('  <ul class="related-links__list">');
  safeItems.forEach((item, index) => {
    const promptAttr = encodeURIComponent(item.prompt);
    const labelId = `related-link-${Date.now()}-${index}`;
    lines.push(
      `<li class="related-links__item" data-prompt="${promptAttr}">` +
        `<span class="related-links__item-title" id="${labelId}">${item.title}</span>` +
        '<button class="related-links__action" type="button" aria-label="Ask this question">' +
        '<span class="material-icons" aria-hidden="true">add</span>' +
        "</button>" +
        "</li>"
    );
    if (index < safeItems.length - 1) {
      lines.push('<li class="related-links__divider" aria-hidden="true"></li>');
    }
  });
  lines.push("  </ul>");
  lines.push("</div>");

  const block = lines.join("");
  $(block).appendTo(".chats").hide().fadeIn(250);
  scrollToBottomOfResults();
  persistChat();
}

/**
 * Render inline chart - displays chart images directly in the chat
 * Used by KB and SQL MCP servers for visualizations
 */
function renderInlineChart(data) {
  if (!data || typeof data !== "object") return;
  
  const imageUrl = data.image_url;
  if (!imageUrl) return;
  
  const alt = escapeHtml(data.alt || "Chart");
  const caption = escapeHtml(data.caption || "");
  
  // Create inline chart HTML
  const chartParts = [
    '<div class="inline-chart">',
    '  <div class="inline-chart__wrapper">',
    '    <img src="' + imageUrl + '" alt="' + alt + '" class="inline-chart__image" loading="lazy">',
    '  </div>',
  ];
  
  if (caption) {
    chartParts.push('  <div class="inline-chart__caption">' + caption + '</div>');
  }
  
  chartParts.push('</div>');
  
  const $chart = $(chartParts.join(""));
  
  // Add click handler to open modal for full-size view
  $chart.find('.inline-chart__image').on('click', function() {
    openChartModal(imageUrl, alt);
  });
  
  // Handle image load errors
  $chart.find('img').on("error", function() {
    const container = $(this).closest(".inline-chart");
    if (container && container.length) {
      container.html('<div class="inline-chart__caption" style="padding: 20px; text-align: center;">⚠️ Chart image could not be loaded</div>');
    }
  });
  
  $chart.appendTo(".chats").hide().fadeIn(400);
  scrollToBottomOfResults();
  persistChat();
}

/**
 * Open modal to view chart in full size
 */
function openChartModal(imageUrl, alt) {
  // Remove existing modal if any
  $('.chart-modal').remove();
  
  const modalHtml = [
    '<div class="chart-modal" onclick="closeChartModal()">',
    '  <span class="chart-modal__close" onclick="closeChartModal()">&times;</span>',
    '  <img class="chart-modal__content" src="' + imageUrl + '" alt="' + (alt || 'Chart') + '">',
    '</div>'
  ].join('');
  
  $(modalHtml).appendTo('body');
  
  // Small delay to trigger CSS transition
  setTimeout(function() {
    $('.chart-modal').addClass('active');
  }, 10);
  
  // Close on escape key
  $(document).one('keydown.chartModal', function(e) {
    if (e.key === 'Escape') {
      closeChartModal();
    }
  });
}

/**
 * Close the chart modal
 */
function closeChartModal() {
  const $modal = $('.chart-modal');
  $modal.removeClass('active');
  setTimeout(function() {
    $modal.remove();
  }, 300);
  $(document).off('keydown.chartModal');
}

const DEFAULT_ADAPTIVE_HOST_CONFIG = {
  fontFamily: "Open Sans, sans-serif",
  spacing: {
    small: 8,
    default: 12,
    medium: 16,
    large: 20,
    extraLarge: 24,
    padding: 16,
  },
  imageSizes: {
    small: 40,
    medium: 80,
    large: 160
  },
  imageSet: {
    imageSize: "medium",
    maxImageHeight: 300
  },
  media: {
    allowInlinePlayback: true
  },
  inputs: {
    label: {
      requiredInputs: {
        weight: "bolder"
      }
    }
  },
  supportsInteractivity: true,
  choiceSet: {
    style: "compact"
  }
};

function renderAdaptiveCardPayload(rawPayload) {
  console.log("🎨 ============= ADAPTIVE CARD RENDERING START =============");
  console.log("🎨 Raw payload received:", JSON.stringify(rawPayload, null, 2));
 
  if (typeof AdaptiveCards === "undefined") {
    console.error("❌ AdaptiveCards library not loaded!");
    const fallbackMsg = "Adaptive card renderer unavailable. Please refresh the page.";
    const BotResponse = getBotResponse(fallbackMsg, fallbackMsg, "ltr");
    $(BotResponse).appendTo(".chats").hide().fadeIn(1000);
    scrollToBottomOfResults();
    persistChat();
    return;
  }

  const payload = rawPayload || {};
 
  // Extract the card data - it might be nested differently
  let cardData = null;
 
  if (payload.data && payload.data.card) {
    // Format: { data: { card: {...}, data: {...} } }
    cardData = payload.data.card;
    console.log("🎨 Found card in payload.data.card");
  } else if (payload.card) {
    // Format: { card: {...} }
    cardData = payload.card;
    console.log("🎨 Found card in payload.card");
  } else if (payload.data) {
    // Format: { data: {...} } (card might be the data itself)
    cardData = payload.data;
    console.log("🎨 Found card in payload.data");
  } else {
    // Format: card is the payload itself
    cardData = payload;
    console.log("🎨 Using payload as card directly");
  }
 
  console.log("🎨 EXTRACTED CARD DATA:", JSON.stringify(cardData, null, 2));
  console.log("🎨 Card type:", typeof cardData);
  console.log("🎨 Card has body:", cardData && cardData.body ? true : false);
  console.log("🎨 Card body length:", cardData && cardData.body ? cardData.body.length : 0);
 
  // Check if card data is empty
  if (!cardData || typeof cardData !== 'object') {
    console.error("❌ Card data is invalid or undefined!");
    console.error("Card data value:", cardData);
    const fallbackMsg = "The adaptive card data is invalid.";
    const BotResponse = getBotResponse(fallbackMsg, fallbackMsg, "ltr");
    $(BotResponse).appendTo(".chats").hide().fadeIn(1000);
    scrollToBottomOfResults();
    persistChat();
    return;
  }
 
  // Check if card has body
  if (!cardData.body || cardData.body.length === 0) {
    console.error("❌ Card has no body items!");
    console.error("Card structure:", cardData);
    const fallbackMsg = "The adaptive card is empty.";
    const BotResponse = getBotResponse(fallbackMsg, fallbackMsg, "ltr");
    $(BotResponse).appendTo(".chats").hide().fadeIn(1000);
    scrollToBottomOfResults();
    persistChat();
    return;
  }
 
  // Extract metadata which contains template info
  const metadata = payload.metadata || (payload.data && payload.data.metadata) || {};
  const messageDirection = (metadata && metadata.direction === 'rtl') ? 'rtl' : 'ltr';
 
  console.log("🎨 EXTRACTED METADATA:", metadata);
  console.log("🎨 TEMPLATE VALUE:", metadata.template);
 
  const hostTheme = payload.hostTheme || (payload.data && payload.data.hostTheme) || null;
  const warnings = Array.isArray(payload.warnings) ? payload.warnings : (payload.data && Array.isArray(payload.data.warnings) ? payload.data.warnings : []);
 
  console.log("🎨 Host theme:", hostTheme);
  console.log("🎨 Warnings:", warnings);

  const adaptiveCard = new AdaptiveCards.AdaptiveCard();

  const applyHostConfigAndRender = (hostConfigJson) => {
    try {
      adaptiveCard.hostConfig = new AdaptiveCards.HostConfig(hostConfigJson || DEFAULT_ADAPTIVE_HOST_CONFIG);
    } catch (err) {
      console.warn("Failed to apply host config, falling back to default.", err);
      adaptiveCard.hostConfig = new AdaptiveCards.HostConfig(DEFAULT_ADAPTIVE_HOST_CONFIG);
    }

    try {
      console.log("🎨 Attempting to parse card data...");
      adaptiveCard.parse(cardData);
      console.log("✅ Adaptive card parsed successfully");
      console.log("🎨 Card title:", cardData.title || "No title");
      console.log("🎨 Card body items:", cardData.body ? cardData.body.length : 0);
     
      // Debug: Check if ChoiceSet inputs are present
      const inputs = adaptiveCard.getAllInputs();
      console.log("🎨 Card inputs found:", inputs.length);
      inputs.forEach((input, index) => {
        console.log(`  Input ${index}:`, {
          id: input.id,
          type: input.constructor.name,
          value: input.value
        });
      });
     
    } catch (parseErr) {
      console.error("❌ Adaptive card parsing failed:", parseErr);
      console.error("❌ Parse error details:", parseErr.message);
      console.error("❌ Card data that failed:", JSON.stringify(cardData, null, 2));
      const fallbackMsg = "I wasn't able to render the adaptive card I generated. Please try again.";
      const BotResponse = getBotResponse(fallbackMsg, fallbackMsg, "ltr");
      $(BotResponse).appendTo(".chats").hide().fadeIn(1000);
      scrollToBottomOfResults();
      persistChat();
      return;
    }

    adaptiveCard.onExecuteAction = function (action) {
      try {
        if (action instanceof AdaptiveCards.SubmitAction) {
          const inputs = {};
          adaptiveCard.getAllInputs().forEach((input) => {
            if (input && input.id) {
              console.log(`Input captured: ${input.id} = ${input.value}`);
              inputs[input.id] = input.value;
            }
          });

          console.log("All captured inputs:", inputs);

          const merged = { ...(action.data || {}), ...inputs };
         
          // Try to get employee_id from multiple sources
          let employeeIdFromCard = null;
         
          // Method 1: Try to get from the card element that triggered this event
          const cardElement = event.target.closest('[data-employee-id]');
          if (cardElement && cardElement.hasAttribute('data-employee-id')) {
            employeeIdFromCard = cardElement.getAttribute('data-employee-id');
            console.log(`Found employee_id from closest card element: ${employeeIdFromCard}`);
          }
         
          // Method 2: Try to get from any card element on the page (fallback)
          if (!employeeIdFromCard) {
            const allCardElements = document.querySelectorAll('[data-employee-id]');
            if (allCardElements.length > 0) {
              // Get the last one (most recent card)
              employeeIdFromCard = allCardElements[allCardElements.length - 1].getAttribute('data-employee-id');
              console.log(`Found employee_id from last card element: ${employeeIdFromCard}`);
            }
          }
         
          // Method 3: Try to get from the adaptive card wrapper
          if (!employeeIdFromCard) {
            const adaptiveCardWrapper = event.target.closest('.adaptive-card-wrapper');
            if (adaptiveCardWrapper) {
              const parentContainer = adaptiveCardWrapper.parentElement;
              if (parentContainer && parentContainer.hasAttribute('data-employee-id')) {
                employeeIdFromCard = parentContainer.getAttribute('data-employee-id');
                console.log(`Found employee_id from wrapper parent: ${employeeIdFromCard}`);
              }
            }
          }
         
          // Add the employee_id to merged if found and not already present
          if (employeeIdFromCard && !merged.employee_id) {
            merged.employee_id = employeeIdFromCard;
            console.log(`Added employee_id to merged data: ${employeeIdFromCard}`);
          }
         
          const submitType = (merged.submit || merged.action || "").toString().toLowerCase();

          console.log("Form submission detected:", {
            submitType: submitType,
            mergedData: merged,
            inputs: inputs,
            actionData: action.data,
            employeeIdFromCard: employeeIdFromCard,
            finalEmployeeId: merged.employee_id
          });

          if (submitType === "contact_info_submit") {
            setUserResponse("Contact information submitted");
            send("/submit_contact_info", {
              person_name: merged.person_name || "",
              phone_number: merged.phone_number || "",
              address: merged.address || "",
            });
          } else if (submitType === "submit_leave_form" || submitType === "leave_form_submit" || submitType === "leave_calculator_submit") {
            // Handle leave form submission - send with / prefix so RASA recognizes it as intent
            setUserResponse("Leave request submitted");
            console.log("Sending leave form submission to RASA:", merged);
           
            // Use employee_id from merged data, which should include card metadata
            let employeeId = merged.employee_id;
            if (!employeeId || employeeId === 'default_user') {
              console.warn("No valid employee_id found in form data or card metadata, using fallback");
              employeeId = '1'; // Only fallback if absolutely no other source
            }
           
            sendLeaveSubmission(employeeId, merged);
          } else {
            console.log("Generic form submission:", submitType);
            const summaryText = merged.summary || merged.title || "Adaptive card submitted";
            setUserResponse(summaryText);
            send(summaryText, {
              source: "adaptive_card",
              data: merged,
            });
          }
        } else if (action instanceof AdaptiveCards.OpenUrlAction) {
          const targetUrl = action.url || (action.data && action.data.url);
          if (targetUrl) {
            window.open(targetUrl, "_blank", "noopener");
          }
        }
      } catch (execErr) {
        console.error("Error executing adaptive card action:", execErr);
      }
    };

    // Helper function to send leave submission
    function sendLeaveSubmission(employeeId, merged) {
      console.log("=== FRONTEND LEAVE SUBMISSION DEBUG ===");
      console.log("Final leave submission data:", {
        employee_id: employeeId,
        start_date: merged.start_date,
        end_date: merged.end_date,
        leave_type: merged.leave_type,
        merged: merged
      });
     
      const submissionPayload = {
        source: "adaptive_card",
        employee_id: employeeId,
        start_date: merged.start_date,
        end_date: merged.end_date,
        leave_type: merged.leave_type,
        data: merged,
      };
     
      console.log("Payload being sent to RASA:", submissionPayload);
     
      // Send the form data in the format expected by RASA action
      send("/submit_leave_form", submissionPayload);
    }

    const renderedCard = adaptiveCard.render();
    if (renderedCard) {
      console.log("✅ Adaptive card rendered successfully");
      console.log("🎨 Rendered card element:", renderedCard);
      console.log("🎨 Rendered card HTML preview:", renderedCard.outerHTML.substring(0, 300));
     
      // Add employee_id as data attribute if present in payload metadata
      if (payload.metadata && payload.metadata.employee_id) {
        renderedCard.setAttribute('data-employee-id', payload.metadata.employee_id);
      }
     
      // Debug: Check rendered DOM for choice sets
      const choiceSets = renderedCard.querySelectorAll('[class*="choice"], select, input[type="radio"], input[type="checkbox"]');
      console.log(`🎨 Found ${choiceSets.length} choice/input elements in rendered card`);
     
      // Debug: Check if card has any visible content
      const textContent = renderedCard.textContent || renderedCard.innerText;
      console.log(`🎨 Card text content length: ${textContent.length}`);
      console.log(`🎨 Card text preview: ${textContent.substring(0, 100)}`);
     
      const $container = $('<div class="bot-adaptive-card"></div>');
      if (messageDirection === 'rtl') {
        $container.addClass('rtl').attr('dir', 'rtl');
      }
     
      // Add template-specific class for CSS styling
      // Get template from metadata - check both 'template' and 'template_id' keys
      const template = metadata.template || metadata.template_id;
      if (template) {
        $container.addClass(`adaptive-card-${template}`);
        console.log(`✅ Added template class: adaptive-card-${template}`);
        console.log(`Container classes:`, $container.attr('class'));
      } else {
        console.warn("⚠️ No template found in metadata, using default styling");
        console.log("Metadata received:", metadata);
      }
     
      // Add employee_id as data attribute to the container as well
      if (metadata && metadata.employee_id) {
        $container.attr('data-employee-id', metadata.employee_id);
        console.log(`Set data-employee-id on container: ${metadata.employee_id}`);
      }
     
      $container.append('<img class="botAvatar" src="./static/images/aliza-icon.jpg"/>');
      const $wrapper = $('<div class="adaptive-card-wrapper"></div>');
      if (messageDirection === 'rtl') {
        $wrapper.addClass('rtl').attr('dir', 'rtl');
        try { renderedCard.setAttribute('dir', 'rtl'); } catch (e) {}
      } else {
        // Fallback: detect Arabic in rendered card text and switch to RTL
        try {
          const cardText = (renderedCard.textContent || renderedCard.innerText || '');
          if (containsArabicChars(cardText)) {
            $container.addClass('rtl').attr('dir', 'rtl');
            $wrapper.addClass('rtl').attr('dir', 'rtl');
            renderedCard.setAttribute('dir', 'rtl');
          }
        } catch (e) {}
      }
     
      // Also add template class to the wrapper (template already extracted above)
      if (template) {
        $wrapper.addClass(`adaptive-card-${template}`);
        console.log(`✅ Added template class to wrapper: adaptive-card-${template}`);
        console.log(`Wrapper classes:`, $wrapper.attr('class'));
      }
     
      // Also add to the wrapper
      if (metadata && metadata.employee_id) {
        $wrapper.attr('data-employee-id', metadata.employee_id);
      }
     
      $wrapper.append(renderedCard);
      $container.append($wrapper);
      $container.append('<div class="clearfix"></div>');
      $container.hide().appendTo(".chats").fadeIn(400);
     
      // DEBUG: Log the actual HTML structure after appending
      console.log("📋 Final HTML structure:");
      console.log($container[0].outerHTML.substring(0, 500)); // First 500 chars
      console.log("Full class list on container:", $container[0].className);
      console.log("🎨 ============= ADAPTIVE CARD RENDERING COMPLETE =============");
     
      // Post-append debugging
      setTimeout(() => {
        const choiceElements = document.querySelectorAll('.chats [class*="choice"], .chats select, .chats input[type="radio"], .chats input[type="checkbox"]');
        console.log(`Found ${choiceElements.length} choice elements in DOM after append:`, choiceElements);
       
        // Check for hidden elements
        choiceElements.forEach((el, index) => {
          const styles = window.getComputedStyle(el);
          console.log(`Choice element ${index} styles:`, {
            display: styles.display,
            visibility: styles.visibility,
            opacity: styles.opacity,
            width: styles.width,
            height: styles.height
          });
        });
      }, 500);
     
      if (warnings.length) {
        const warningText = warnings.slice(0, 2).join(" | ");
        const warningNode = getBotResponse(`⚠️ ${warningText}`, `⚠️ ${warningText}`, "ltr");
        $(warningNode).appendTo(".chats").hide().fadeIn(400);
      }
      scrollToBottomOfResults();
      persistChat();
    } else {
      console.error("❌ Adaptive card render() returned null or undefined");
      console.error("Card parse was successful but render failed");
      console.error("This might indicate an issue with the AdaptiveCards library");
    }
  };

  if (hostTheme) {
    fetch(`/static/hostconfigs/${hostTheme}.json`, { cache: "no-cache" })
      .then((resp) => (resp.ok ? resp.json() : Promise.reject(new Error("Failed to load host config"))))
      .then((cfg) => applyHostConfigAndRender(cfg))
      .catch((err) => {
        console.warn(`Failed to fetch host config "${hostTheme}", using default.`, err);
        applyHostConfigAndRender(DEFAULT_ADAPTIVE_HOST_CONFIG);
      });
  } else {
    applyHostConfigAndRender(DEFAULT_ADAPTIVE_HOST_CONFIG);
  }
}

/**
 * Set user response on the chat screen
 * @param {String} message user message
 */
function setUserResponse(message) {
  const direction = detectTextDirection(message);
  const directionClass = direction === 'rtl' ? ' rtl' : '';
  const directionAttr = direction === 'rtl' ? " dir='rtl'" : '';
  const user_response = `<img class="userAvatar" src='./static/images/userAvatar.jpg'><p class="userMsg${directionClass}"${directionAttr}>${message} </p><div class="clearfix"></div>`;
  $(user_response).appendTo(".chats").show("slow");
 
  $(".usrInput").val("");
  applyInputDirectionFromValue();
  scrollToBottomOfResults();
  showBotTyping();
  $(".suggestions").remove();
  persistChat();
}
 
/**
 * returns formatted bot response
 * @param {String} text bot message response's text
 *
 */
function cleanMessageText(text) {
  if (!text) return '';
  let author = '';
  const dashMatch = text.match(/\s*[-–—]\s*(.+)$/); // captures -, – or —
  if (dashMatch) {
    author = dashMatch[1].trim();
    text = text.replace(/\s*[-–—]\s*(.+)$/, '');
  }

  let cleanText = text
    // Remove leaked reasoning tags and any HTML-like content that could break attributes
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<\/?[a-zA-Z][^>]*>/g, '')
    .replace(/["“”]/g, '')
    .replace(/([\u2700-\u27BF]|[\uE000-\uF8FF]|\u24C2|[\uD83C-\uDBFF\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|\uD83E[\uDD00-\uDDFF])/g, '')
    .replace(/•|\n|•\s*/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/\*\*/g, '')
    .replace(/[<>]/g, '')
    .trim();

  if (author) {
    cleanText = `${cleanText}. ${author}`;
  }
  return cleanText;
}

function getBotResponse(textOrHtml, rawTextForControls = null, direction = 'ltr') {

  const safeTextForAttr = rawTextForControls ? encodeURIComponent(cleanMessageText(rawTextForControls)) : '';

  const plainTextForAudio = rawTextForControls ? cleanMessageText(rawTextForControls) : '';



  const controls = rawTextForControls

    ? ` <button class="play-btn" data-playing="false" data-text="${plainTextForAudio}" title="Play or stop audio" aria-label="Play or stop audio"><i class="fa fa-volume-up" aria-hidden="true"></i></button>

         <button class="avatar-btn" data-message="${safeTextForAttr}" title="Generate avatar" aria-label="Generate avatar"><i class="fa fa-user-circle" aria-hidden="true"></i></button>`

    : '';



  const content = `${textOrHtml}${controls}`;

  const directionClass = direction === 'rtl' ? ' rtl' : '';

  const directionAttr = direction === 'rtl' ? " dir='rtl'" : '';

  const botResponse = `<img class="botAvatar" src="./static/images/aliza-icon.jpg"/><span class="botMsg${directionClass}"${directionAttr}>${content}</span><div class="clearfix"></div>`;

  return botResponse;

}



/**
 * renders bot response on to the chat screen
 * @param {Array} response json array containing different types of bot response
 *
 * for more info: `https://rasa.com/docs/rasa/connectors/your-own-website#request-and-response-format`
 */
function setBotResponse(response) {
  // renders bot response after 500 milliseconds
  setTimeout(() => {
    hideBotTyping();
    if (response.length < 1) {
      // if there is no response from Rasa, send  fallback message to the user
      const fallbackMsg = "I am facing some issues, please try again later!!!";
 
      const BotResponse = getBotResponse(fallbackMsg, fallbackMsg, 'ltr');

      $(BotResponse).appendTo(".chats").hide().fadeIn(1000);
      scrollToBottomOfResults();
      persistChat();
    } else {
      // if we get response from Rasa
      // Ensure avatar modal exists for upcoming interactions
      ensureAvatarModalExists();
      for (let i = 0; i < response.length; i += 1) {
        const direction = (response[i] && response[i].metadata && response[i].metadata.direction === 'rtl') ? 'rtl' : 'ltr';
        // check if the response contains "text"
        if (Object.hasOwnProperty.call(response[i], "text")) {
          if (response[i].text != null) {
            // convert the text to mardown format using showdown.js(https://github.com/showdownjs/showdown);
            let botResponse;
            let html = converter.makeHtml(response[i].text);
            html = html
              .replaceAll("<p>", "")
              .replaceAll("</p>", "")
              .replaceAll("<strong>", "<b>")
              .replaceAll("</strong>", "</b>");
            html = html.replace(/(?:\r\n|\r|\n)/g, "<br>");
            // console.log(html);
            // check for blockquotes
            if (html.includes("<blockquote>")) {
              html = html.replaceAll("<br>", "");
              botResponse = getBotResponse(html, response[i].text, direction);
            }
            // check for image
            if (html.includes("<img")) {
              html = html.replaceAll("<img", '<img class="imgcard_mrkdwn" ');
              botResponse = getBotResponse(html, response[i].text, direction);
            }
            // check for preformatted text
            if (html.includes("<pre") || html.includes("<code>")) {
              botResponse = getBotResponse(html, response[i].text, direction);
            }
            // check for list text
            if (
              html.includes("<ul") ||
              html.includes("<ol") ||
              html.includes("<li") ||
              html.includes("<h3")
            ) {
              html = html.replaceAll("<br>", "");
              // botResponse = `<img class="botAvatar" src="./static/images/aliza-icon.jpg"/><span class="botMsg">${html}</span><div class="clearfix"></div>`;
              botResponse = getBotResponse(html, response[i].text, direction);
            }

            if (!botResponse) {
              const fallbackHtml = html && html.length ? html : escapeHtml(response[i].text);
              botResponse = getBotResponse(fallbackHtml, response[i].text, direction);
            }
            // append the bot response on to the chat screen
            $(botResponse).appendTo(".chats").hide().fadeIn(1000);
            // Bind media controls for TTS and Avatar
            bindPlayButtons();
            bindAvatarButtons();
            persistChat();
          }
        }
 
        // check if the response contains "images"
        if (Object.hasOwnProperty.call(response[i], "image")) {
          if (response[i].image !== null) {
            const BotResponse = `<div class="singleCard"><img class="imgcard" src="${response[i].image}"></div><div class="clearfix"></div>`;
 
            $(BotResponse).appendTo(".chats").hide().fadeIn(1000);
            persistChat();
          }
        }
 
        // check if the response contains "buttons"
        if (Object.hasOwnProperty.call(response[i], "buttons")) {
          if (response[i].buttons.length > 0) {
            addSuggestion(response[i].buttons);
          }
        }
 
        // check if the response contains "attachment"
        if (Object.hasOwnProperty.call(response[i], "attachment")) {
          if (response[i].attachment != null) {
            if (response[i].attachment.type === "video") {
              // check if the attachment type is "video"
              const video_url = response[i].attachment.payload.src;
 
              const BotResponse = `<div class="video-container"> <iframe src="${video_url}" frameborder="0" allowfullscreen></iframe> </div>`;
              $(BotResponse).appendTo(".chats").hide().fadeIn(1000);
            persistChat();
            }
          }
        }
        // check if the response contains "custom" message
        if (Object.hasOwnProperty.call(response[i], "custom")) {
          const { payload } = response[i].custom;
          if (payload === "quickReplies") {
            // check if the custom payload type is "quickReplies"
            const quickRepliesData = response[i].custom.data;
            showQuickReplies(quickRepliesData);
            return;
          }
 
          // check if the custom payload type is "pdf_attachment"
          if (payload === "pdf_attachment") {
            renderPdfAttachment(response[i]);
            return;
          }

          if (payload === "related_links") {
            renderRelatedLinks(response[i].custom.data || {});
            continue;
          }

          if (payload === "visual_panel") {
            renderVisualPanel(response[i].custom.data || {});
            continue;
          }
          
          // Handle inline charts from KB/SQL MCP servers
          if (payload === "inline_chart") {
            renderInlineChart(response[i].custom.data || {});
            continue;
          }
 
          // check if the custom payload type is "dropDown"
          if (payload === "dropDown") {
            const dropDownData = response[i].custom.data;
            renderDropDwon(dropDownData);
            return;
          }
 
          // check if the custom payload type is "location"
          if (payload === "location") {
            $(".usrInput").prop("disabled", true);
            getLocation();
            scrollToBottomOfResults();
            return;
          }
 
          // check if the custom payload type is "cardsCarousel"
          if (payload === "cardsCarousel") {
            const restaurantsData = response[i].custom.data;
            showCardsCarousel(restaurantsData);
            return;
          }
 
          // check if the custom payload type is "chart"
          if (payload === "chart") {
            /**
             * sample format of the charts data:
             *  var chartData =  { "title": "Leaves", "labels": ["Sick Leave", "Casual Leave", "Earned Leave", "Flexi Leave"], "backgroundColor": ["#36a2eb", "#ffcd56", "#ff6384", "#009688", "#c45850"], "chartsData": [5, 10, 22, 3], "chartType": "pie", "displayLegend": "true" }
             */
 
            const chartData = response[i].custom.data;
            const {
              title,
              labels,
              backgroundColor,
              chartsData,
              chartType,
              displayLegend,
            } = chartData;
 
            // pass the above variable to createChart function
            createChart(
              title,
              labels,
              backgroundColor,
              chartsData,
              chartType,
              displayLegend
            );
 
            // on click of expand button, render the chart in the charts modal
            $(document).on("click", "#expand", () => {
              createChartinModal(
                title,
                labels,
                backgroundColor,
                chartsData,
                chartType,
                displayLegend
              );
            });
            return;
          }
 
          // check of the custom payload type is "collapsible"
          if (payload === "collapsible") {
            const { data } = response[i].custom;
            // pass the data variable to createCollapsible function
            createCollapsible(data);
          }
 
          // check if the custom payload type is "adaptiveCard"
          if (payload === "adaptiveCard") {
            // Pass the entire custom object so we can access both data and metadata
            renderAdaptiveCardPayload(response[i].custom || {});
            continue;
          }
        }
      }
      scrollToBottomOfResults();
    }
    $(".usrInput").focus();
    persistChat();
    
    // Refresh session sidebar to show updated chat history
    if (typeof window.loadSessions === 'function') {
      window.loadSessions();
    }
  }, 500);
}

// Restore chat when chat widget loads (admin return only)
if (typeof $ !== 'undefined') {
 
async function initializeChatSession() {
  await restoreChatFromStorage();
  await autoGreetIfNeeded();
  applyInputDirectionFromValue();
}

$(document).ready(function(){ initializeChatSession(); });
} else {
  document.addEventListener('DOMContentLoaded', initializeChatSession);
}

// --- Avatar Modal and TTS/Avatar Controls Integration ---
function ensureAvatarModalExists() {
  if ($('#avatarModal').length === 0) {
    const modalHtml = `
      <div id="avatarModal" class="avatar-modal">
        <div class="avatar-modal-content">
          <div class="avatar-modal-header">
            <h3>Alizha Avatar</h3>
            <span class="avatar-modal-close">&times;</span>
          </div>
          <div class="avatar-modal-body">
            <div id="avatarLoader" class="avatar-loader">
              <div class="loading-spinner"></div>
              <p>Generating Avatar Video...</p>
              <div class="loading-progress"><div class="progress-bar"></div></div>
            </div>
            <video id="avatarVideo" controls style="display: none; width: 100%; max-height: 400px; border-radius: 8px;">
              Your browser does not support the video tag.
            </video>
            <div id="avatarError" class="avatar-error" style="display: none;">
              <p>❌ Failed to generate avatar video</p>
            </div>
          </div>
        </div>
      </div>`;
    $('body').append(modalHtml);
    addAvatarModalStyles();
    bindModalCloseEvents();
  }
}

function addAvatarModalStyles() {
  if ($('#avatarModalStyles').length === 0) {
    const styles = `
      <style id="avatarModalStyles">
        .avatar-modal {display:none;position:fixed;z-index:10000;right:0;top:0;width:450px;height:100vh;background-color:rgba(0,0,0,0.8);animation:slideInRight 0.3s ease-out;}
        @keyframes slideInRight {from{transform:translateX(100%);}to{transform:translateX(0);}}
        .avatar-modal-content {background-color:#fff;margin:20px;padding:0;border-radius:12px;box-shadow:0 8px 32px rgba(0,0,0,0.3);height:calc(100vh - 40px);display:flex;flex-direction:column;overflow:hidden;}
        .avatar-modal-header {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:20px;display:flex;justify-content:space-between;align-items:center;}
        .avatar-modal-header h3 {margin:0;font-size:1.4em;font-weight:600;}
        .avatar-modal-close {font-size:28px;font-weight:bold;cursor:pointer;line-height:1;transition:opacity 0.2s;}
        .avatar-modal-close:hover {opacity:0.7;}
        .avatar-modal-body {flex:1;padding:30px;display:flex;flex-direction:column;justify-content:center;align-items:center;background-color:#f8f9fa;}
        .avatar-loader {text-align:center;color:#666;}
        .loading-spinner {width:60px;height:60px;border:4px solid #f3f3f3;border-top:4px solid #667eea;border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 20px;}
        @keyframes spin {0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}
        .loading-progress {width:200px;height:6px;background-color:#e0e0e0;border-radius:3px;margin:20px auto;overflow:hidden;}
        .progress-bar {height:100%;background:linear-gradient(90deg,#667eea,#764ba2);width:0%;animation:progressAnimation 3s ease-in-out infinite;border-radius:3px;}
        @keyframes progressAnimation {0%{width:0%;}50%{width:70%;}100%{width:100%;}}
        .avatar-error {text-align:center;color:#e74c3c;font-size:1.1em;}
        @media(max-width:768px){.avatar-modal{width:100vw;right:0;}.avatar-modal-content{margin:10px;height:calc(100vh - 20px);}}
      </style>`;
    $('head').append(styles);
  }
}

function bindModalCloseEvents() {
  $(document).on('click', '.avatar-modal-close', function() {
    $('#avatarModal').hide();
    const v = $('#avatarVideo')[0];
    if (v) v.pause();
  });
  $(document).on('click', '#avatarModal', function(e) {
    if (e.target === this) {
      $('#avatarModal').hide();
      const v = $('#avatarVideo')[0];
      if (v) v.pause();
    }
  });
  $(document).on('keydown', function(e) {
    if (e.key === 'Escape' && $('#avatarModal').is(':visible')) {
      $('#avatarModal').hide();
      const v = $('#avatarVideo')[0];
      if (v) v.pause();
    }
  });
}

function bindAvatarButtons() {
  $('.avatar-btn').off('click').on('click', function() {
    const $btn = $(this);
    const originalText = $btn.text();
    $btn.text('⏳ Generating...');

    $('#avatarModal').show();
    $('#avatarLoader').show();
    $('#avatarVideo').hide();
    $('#avatarError').hide();

    const xhr = new XMLHttpRequest();
    xhr.open('GET', 'http://127.0.0.1:5002/generate-avatar', true);
    xhr.responseType = 'json';
    xhr.timeout = 4 * 60 * 60 * 1000;

    xhr.onload = function() {
      $btn.text(originalText);
      if (xhr.status === 200) {
        const response = xhr.response;
        if (response && response.video_url) {
          const videoUrl = "http://127.0.0.1:5002" + response.video_url;
          $('#avatarLoader').hide();
          const videoEl = $('#avatarVideo')[0];
          if (videoEl) {
            videoEl.src = videoUrl;
            $('#avatarVideo').show();
            videoEl.play().catch(e => console.log('Auto-play prevented:', e));
          }
        } else {
          $('#avatarLoader').hide();
          $('#avatarError').show();
        }
      } else {
        $('#avatarLoader').hide();
        $('#avatarError').show();
      }
    };

    xhr.ontimeout = function() {
      console.error('Avatar generation timed out');
      $btn.text(originalText);
      $('#avatarLoader').hide();
      $('#avatarError').show();
      $('#avatarError p').text('⏰ Request timed out. Please try again.');
    };

    xhr.onerror = function() {
      console.error('Avatar generation error');
      $btn.text(originalText);
      $('#avatarLoader').hide();
      $('#avatarError').show();
      $('#avatarError p').text('❌ Network error occurred. Please try again.');
    };

    xhr.send();
  });
}

function bindPlayButtons() {
  $('.play-btn').off('click').on('click', async function () {
    const $btn = $(this);
    const encodedText = encodeURIComponent($btn.attr('data-text'));
    $btn.text('⏳');

    try {
      const response = await fetch(`http://127.0.0.1:5001/generate-audio?text=${encodedText}`);
      const data = await response.json();
      if (data && data.audio_file) {
        const audio = new Audio(`http://127.0.0.1:5001${data.audio_file}`);
        audio.play();
        $btn.text('⏸️');
        audio.onended = () => { $btn.text('▶️'); };
      } else {
        console.error('Audio not available:', data);
        $btn.text('❌');
      }
    } catch (error) {
      console.error('Error generating audio:', error);
      $btn.text('❌');
    }
  });
}

// Optional: future-proof helper to add messages with controls (not wired into flow yet)
function addMessageToChat(message) {
  const messageClass = message.sender === 'user' ? 'user-message' : 'bot-message';
  const timestamp = new Date(message.timestamp || Date.now()).toLocaleTimeString();

  function localClean(text){ return cleanMessageText(text); }

  if (message.sender === 'bot' && message.message) {
    const html = getBotResponse(message.message, message.message, detectTextDirection(message.message));
    $(html).appendTo('.chats');
    ensureAvatarModalExists();
    bindPlayButtons();
    bindAvatarButtons();
    scrollToBottomOfResults();
    return;
  }

  // Fallback generic append for user messages
  const direction = detectTextDirection(message.message);
  const directionClass = direction === 'rtl' ? ' rtl' : '';
  const directionAttr = direction === 'rtl' ? " dir='rtl'" : '';
  const user_response = `<img class="userAvatar" src='./static/images/userAvatar.jpg'><p class="userMsg${directionClass}"${directionAttr}>${message.message}</p><div class="clearfix"></div>`;
  $(user_response).appendTo('.chats').show('slow');
  scrollToBottomOfResults();
}
 
/**
 * sends the user message to the rasa server,
 * @param {String} message user message
 */
async function send(message, metadata = null) {
  const data = { message, sender: sender_id };

  const messageDirection = detectTextDirection(message);
 
  // Include user's preferred model in metadata
  const preferredModel = window.getCurrentModelPreference ? window.getCurrentModelPreference() : 'auto';
  const enhancedMetadata = {
    preferred_model: preferredModel === 'auto' ? null : preferredModel,
    ...metadata
  };
  if (!('text_direction' in enhancedMetadata) || !enhancedMetadata.text_direction) {
    enhancedMetadata.text_direction = messageDirection;
  }
 
  if (enhancedMetadata && Object.keys(enhancedMetadata).length > 0) {
    data.metadata = enhancedMetadata;
  }
  await new Promise((r) => setTimeout(r, 2000));
  $.ajax({
    url: rasa_server_url,
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify(data),
    success(botResponse, status) {
      // console.log("Response from Rasa: ", botResponse, "\nStatus: ", status);
 
      // if user wants to restart the chat and clear the existing chat contents
      if (message.toLowerCase() === "/restart") {
        $(".usrInput").prop("disabled", false);
 
        // if you want the bot to start the conversation after restart
        // customActionTrigger();
        return;
      }
      setBotResponse(botResponse);
    },
    error(xhr, textStatus) {
      if (message.toLowerCase() === "/restart") {
        $("#userInput").prop("disabled", false);
        // if you want the bot to start the conversation after the restart action.
        // actionTrigger();
        // return;
      }
 
      // if there is no response from rasa server, set error bot response
      setBotResponse("");
      // console.log("Error from bot end: ", textStatus);
    },
  });
}
 
/**
 * Handle file upload functionality
 * @param {File} file - The file to upload
 */
function handleFileUpload(file) {
  if (!file) return;
 
  // Show upload message
  const uploadMsg = `📤 Uploading ${file.name}...`;
  const uploadResponse = `<img class="userAvatar" src='./static/images/userAvatar.jpg'><p class="userMsg">${uploadMsg}</p><div class="clearfix"></div>`;
  $(uploadResponse).appendTo(".chats").show("slow");
  scrollToBottomOfResults();
  showBotTyping();
  persistChat();
 
  // Determine file type
  const ext = file.name.split('.').pop().toLowerCase();
  const isImage = ['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(ext);
  const isDoc = ['pdf', 'docx', 'txt'].includes(ext);
 
  if (!isImage && !isDoc) {
    hideBotTyping();
    const errorResponse = `<img class="botAvatar" src="./static/images/aliza-icon.jpg"/><p class="botMsg">❌ Unsupported file type. Please upload images (JPG, PNG, GIF, WebP) or documents (PDF, DOCX, TXT).</p><div class="clearfix"></div>`;
    $(errorResponse).appendTo(".chats").hide().fadeIn(1000);
    scrollToBottomOfResults();
    return;
  }
 
  const question = isImage
    ? 'Describe this image in detail'
    : 'Summarize the main points of this document';
 
  const formData = new FormData();
  formData.append('file', file);
  formData.append('file_type', isImage ? 'image' : 'document');
  formData.append('question', question);
  // Add documents to knowledge base for future querying
  formData.append('add_to_kb', isDoc ? 'true' : 'false');
 
  // Upload to Flask backend
  $.ajax({
    url: '/upload',
    method: 'POST',
    data: formData,
    processData: false,
    contentType: false,
    success: (resp) => {
      hideBotTyping();
      if (resp.success) {
        // Show success message
        const successResponse = `<img class="botAvatar" src="./static/images/aliza-icon.jpg"/><p class="botMsg">✅ ${file.name} processed successfully!</p><div class="clearfix"></div>`;
        $(successResponse).appendTo(".chats").hide().fadeIn(1000);
       
        // Show analysis result
        const analysisResponse = `<img class="botAvatar" src="./static/images/aliza-icon.jpg"/><p class="botMsg">${resp.result}</p><div class="clearfix"></div>`;
        $(analysisResponse).appendTo(".chats").hide().fadeIn(1000);
      } else {
        const errorResponse = `<img class="botAvatar" src="./static/images/aliza-icon.jpg"/><p class="botMsg">❌ Upload failed: ${resp.error || 'Unknown error'}</p><div class="clearfix"></div>`;
        $(errorResponse).appendTo(".chats").hide().fadeIn(1000);
      }
      scrollToBottomOfResults();
      persistChat();
    },
    error: (xhr) => {
      hideBotTyping();
      const errorMsg = xhr.responseJSON ? xhr.responseJSON.error : 'Network/server error';
      const errorResponse = `<img class="botAvatar" src="./static/images/aliza-icon.jpg"/><p class="botMsg">❌ Upload failed: ${errorMsg}</p><div class="clearfix"></div>`;
      $(errorResponse).appendTo(".chats").hide().fadeIn(1000);
      scrollToBottomOfResults();
      persistChat();
    }
  });
}
 
/**
 * sends an event to the bot,
 *  so that bot can start the conversation by greeting the user
 *
 * `Note: this method will only work in Rasa 1.x`
 */
// eslint-disable-next-line no-unused-vars
function actionTrigger() {
  $.ajax({
    url: `http://localhost:5005/conversations/${sender_id}/execute`,
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      name: action_name,
      policy: "MappingPolicy",
      confidence: "0.98",
    }),
    success(botResponse, status) {
      // console.log("Response from Rasa: ", botResponse, "\nStatus: ", status);
 
      if (Object.hasOwnProperty.call(botResponse, "messages")) {
        setBotResponse(botResponse.messages);
      }
      $("#userInput").prop("disabled", false);
    },
    error(xhr, textStatus) {
      // if there is no response from rasa server
      setBotResponse("");
      // console.log("Error from bot end: ", textStatus);
      $("#userInput").prop("disabled", false);
    },
  });
}
 
/**
 * sends an event to the custom action server,
 *  so that bot can start the conversation by greeting the user
 *
 * Make sure you run action server using the command
 * `rasa run actions --cors "*"`
 *
 * `Note: this method will only work in Rasa 2.x`
 */
// eslint-disable-next-line no-unused-vars
function customActionTrigger() {
  $.ajax({
    url: "http://localhost:5055/webhook/",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      next_action: action_name,
      tracker: {
        sender_id,
      },
    }),
    success(botResponse, status) {
      // console.log("Response from Rasa: ", botResponse, "\nStatus: ", status);
 
      if (Object.hasOwnProperty.call(botResponse, "responses")) {
        setBotResponse(botResponse.responses);
      }
      $("#userInput").prop("disabled", false);
    },
    error(xhr, textStatus) {
      // if there is no response from rasa server
      setBotResponse("");
      // console.log("Error from bot end: ", textStatus);
      $("#userInput").prop("disabled", false);
    },
  });
}
 
/**
 * clears the conversation from the chat screen
 * & sends the `/resart` event to the Rasa server
 */
function restartConversation() {
  $(".usrInput").prop("disabled", true);
  // destroy the existing chart
  $(".collapsible").remove();
 
  if (typeof chatChart !== "undefined") {
    chatChart.destroy();
  }
 
  $(".chart-container").remove();
  if (typeof modalChart !== "undefined") {
    modalChart.destroy();
  }
  $(".chats").html("");
  $(".usrInput").val("");
  try { sessionStorage.removeItem('autoGreetingShown'); } catch (e) {}
  // Clear server-side session (this user only) then restart Rasa tracker
  $.ajax({
    url: '/clear_chat',
    type: 'POST',
    success: () => {
      try { sessionStorage.removeItem('chat_html'); } catch (e) {}
      send("/restart");
    },
    error: () => {
      // Even if clearing session fails, still attempt restart of Rasa tracker
      try { sessionStorage.removeItem('chat_html'); } catch (e) {}
      send("/restart");
    }
  });
}
// triggers restartConversation function.
$("#restart").click(() => {
  restartConversation();
});
 
/**
 * if user hits enter or send button
 * */
$(".usrInput").on("input", applyInputDirectionFromValue);
$(".usrInput").on("keyup keypress", (e) => {
  applyInputDirectionFromValue();
  const keyCode = e.keyCode || e.which;
 
  const text = $(".usrInput").val();
  if (keyCode === 13) {
    if (text === "" || $.trim(text) === "") {
      e.preventDefault();
      return false;
    }
    // destroy the existing chart, if yu are not using charts, then comment the below lines
    $(".collapsible").remove();
    $(".dropDownMsg").remove();
    if (typeof chatChart !== "undefined") {
      chatChart.destroy();
    }
 
    $(".chart-container").remove();
    if (typeof modalChart !== "undefined") {
      modalChart.destroy();
    }
 
    $("#paginated_cards").remove();
    $(".suggestions").remove();
    $(".quickReplies").remove();
    $(".usrInput").blur();
    setUserResponse(text);
    send(text);
    e.preventDefault();
    return false;
  }
  return true;
});
 
$("#sendButton").on("click", (e) => {
  const text = $(".usrInput").val();
  if (text === "" || $.trim(text) === "") {
    e.preventDefault();
    return false;
  }
  // destroy the existing chart
  if (typeof chatChart !== "undefined") {
    chatChart.destroy();
  }
 
  $(".chart-container").remove();
  if (typeof modalChart !== "undefined") {
    modalChart.destroy();
  }
 
  $(".suggestions").remove();
  $("#paginated_cards").remove();
  $(".quickReplies").remove();
  $(".usrInput").blur();
  $(".dropDownMsg").remove();
  setUserResponse(text);
  send(text);
  e.preventDefault();
  return false;
});
 
// File upload event handler
$("#file-input").on("change", function(e) {
  const file = e.target.files[0];
  if (file) {
    handleFileUpload(file);
    // Clear the input so the same file can be uploaded again
    $(this).val('');
  }
  // remove any flash state on attach button
  $(".attach-btn").removeClass('attach-flash').blur();
  persistChat();
});
 
// Attach button click handler
$(".attach-btn").on("click", function(e) {
  e.preventDefault();
  const $btn = $(this);
  $btn.addClass('attach-flash');
  setTimeout(() => { $btn.removeClass('attach-flash'); }, 1200);
  $btn.blur();
  $("#file-input").click();
});

$(document).off("click.relatedLinks", ".related-links__action").on("click.relatedLinks", ".related-links__action", function (event) {
  event.preventDefault();
  const parent = $(this).closest(".related-links__item");
  if (!parent.length) return;
  const encodedPrompt = parent.data("prompt");
  if (!encodedPrompt) return;
  let prompt = "";
  try {
    prompt = decodeURIComponent(encodedPrompt);
  } catch (err) {
    prompt = encodedPrompt;
  }
  prompt = (prompt || "").trim();
  if (!prompt) return;
  const container = parent.closest(".related-links");
  if (container.length) {
    container.remove();
    persistChat();
  }
  setUserResponse(prompt);
  send(prompt, { source: "related_link" });
});

 
/**
* -----------------------------------------------------------------
* WHISPER.CPP SPEECH-TO-TEXT INTEGRATION
* -----------------------------------------------------------------
*/
let mediaRecorder;
let audioChunks = [];
let recognitionActive = false;
 
const micBtn = document.getElementById("mic-btn");
const userInput = $(".usrInput");
 
// --- Main function to start recording ---
async function startRecording() {
    if (recognitionActive) return;
 
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = []; // Clear previous recording
 
        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };
 
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            sendAudioToServer(audioBlob);
            stream.getTracks().forEach(track => track.stop()); // Release microphone
        };
 
        mediaRecorder.start();
        recognitionActive = true;
        micBtn.classList.add("is-recording");
        userInput.attr("placeholder", "Listening... Click to stop.");
 
    } catch (error) {
        console.error("Mic access denied:", error);
        userInput.attr("placeholder", "Mic access denied.");
    }
}
 
// --- Main function to stop recording ---
function stopRecording() {
    if (!recognitionActive || !mediaRecorder) return;
    mediaRecorder.stop();
    recognitionActive = false;
    micBtn.classList.remove("is-recording");
    userInput.attr("placeholder", "Processing...");
}
 
// --- Function to send audio to the server ---
async function sendAudioToServer(audioBlob) {
    const serverAddress = "ec2-3-7-195-243.ap-south-1.compute.amazonaws.com";
    const serverUrl = `http://${serverAddress}:8888/transcribe`;
 
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
 
    try {
        const response = await fetch(serverUrl, {
            method: 'POST',
            body: formData,
        });
 
        if (response.ok) {
            const result = await response.json();
            // The Whisper.cpp output includes timestamps and other info, so we just grab the text
            const cleanText = result.text.replace(/\[.*?\]/g, '').trim();
            userInput.val(userInput.val() + cleanText + " ");
        } else {
            console.error("Server failed to process audio.");
            userInput.val("Sorry, I couldn't understand that.");
        }
    } catch (error) {
        console.error("Error sending audio:", error);
        userInput.val("Error connecting to STT server.");
    } finally {
        userInput.attr("placeholder", "Type a message...");
    }
}
 
// --- Event Listener for the Microphone Button ---
micBtn.addEventListener("click", () => {
    if (recognitionActive) {
        stopRecording();
    } else {
        startRecording();
    }
});
