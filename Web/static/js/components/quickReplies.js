/**
 * shows the quick replies
 * @param {Array} quickRepliesData quick replies data
 */
function showQuickReplies(quickRepliesData) {
    const quickRepliesDiv = `<div class="quickReplies"></div>`;
    $(quickRepliesDiv).appendTo(".chats");
   
    quickRepliesData.forEach((quickReply) => {
      const quickReplyChip = `<div class="chip" data-payload="${quickReply.payload}">${quickReply.title}</div>`;
      $(quickReplyChip).appendTo(".quickReplies");
    });
   
    scrollToBottomOfResults();
  }
   
  // Handle quick reply clicks
  $(document).on("click", ".quickReplies .chip", function() {
    const payload = $(this).data("payload");
    const title = $(this).text();
   
    // Remove quick replies after clicking
    $(".quickReplies").remove();
   
    // Show user message
    setUserResponse(title);
   
    // Send payload to RASA
    send(payload);
  });