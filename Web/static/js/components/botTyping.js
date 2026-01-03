/**
 * shows the bot typing indicator
 */
function showBotTyping() {
    const botTyping = `<div class="botTyping"><img class="botAvatar" src="./static/images/aliza-icon.jpg"/><div class="typing-indicator"><div class="typing-dots"><span></span><span></span><span></span></div></div></div>`;
    $(botTyping).appendTo(".chats");
    scrollToBottomOfResults();
  }
   
  /**
   * hides the bot typing indicator
   */
  function hideBotTyping() {
    $(".botTyping").remove();
  }