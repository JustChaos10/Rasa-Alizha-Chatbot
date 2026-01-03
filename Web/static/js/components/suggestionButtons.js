/**
 * adds the suggestion buttons to the chat screen
 * @param {Array} suggestions buttons array
 */
function addSuggestion(suggestions) {
    setTimeout(() => {
      const suggestionDiv = `<div class="suggestions"></div>`;
      $(suggestionDiv).appendTo(".chats");
     
      // Loop through suggestions and add buttons
      suggestions.forEach((suggestion) => {
        const suggestionButton = `<button class="suggestion-btn" data-payload="${suggestion.payload}">${suggestion.title}</button>`;
        $(suggestionButton).appendTo(".suggestions");
      });
     
      scrollToBottomOfResults();
    }, 1000);
  }
   
  // Handle suggestion button clicks
  $(document).on("click", ".suggestion-btn", function() {
    const payload = $(this).data("payload");
    const title = $(this).text();
   
    // Remove all suggestions after clicking
    $(".suggestions").remove();
   
    // Show user message
    setUserResponse(title);
   
    // Send payload to RASA
    send(payload);
  });