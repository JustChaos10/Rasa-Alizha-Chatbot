/* module for importing other js files */
function include(file) {
    const script = document.createElement('script');
    script.src = file;
    script.type = 'text/javascript';
    script.defer = true;
   
    document.getElementsByTagName('head').item(0).appendChild(script);
  }
   
  // Bot pop-up intro
  document.addEventListener("DOMContentLoaded", () => {
    const elemsTap = document.querySelector(".tap-target");
    if (elemsTap && typeof M !== 'undefined') {
      // eslint-disable-next-line no-undef
      const instancesTap = M.TapTarget.init(elemsTap, {});
      instancesTap.open();
      setTimeout(() => {
        instancesTap.close();
      }, 4000);
    }
  });
   
  /* import components */
  include('./static/js/components/index.js');
   
window.addEventListener('load', () => {
  const DEBUG = !!window.DEBUG_WIDGET;
  const WIDGET_STATE_KEY = 'widgetOpen';

  function setState(open) {
    sessionStorage.setItem(WIDGET_STATE_KEY, open ? 'true' : 'false');
    if (DEBUG) console.log('[Widget] state ->', open);
  }

  function isOpen() {
    return document.querySelector('.widget')?.classList.contains('is-open');
  }

  function openWidget(e) {
    if (e) e.preventDefault();
    $('.widget').addClass('is-open');
    setState(true);
  }

  function closeWidget(e) {
    if (e) e.preventDefault();
    $('.widget').removeClass('is-open');
    setState(false);
    if (typeof scrollToBottomOfResults === 'function') scrollToBottomOfResults();
  }
  
  function toggleWidget(e) {
    if (e) e.preventDefault();
    if ($('.widget').hasClass('is-open')) {
      closeWidget();
    } else {
      openWidget();
    }
  }
    // initialization
    $(document).ready(() => {
      // Bot pop-up intro
      $("div").removeClass("tap-target-origin");
   
      // drop down menu for close, restart conversation & clear the chats.
      if (typeof M !== 'undefined') {
        $(".dropdown-trigger").dropdown();
        // initiate the modal for displaying the charts,
        // if you dont have charts, then you comment the below line
        $(".modal").modal();
      }
   
      // enable this if u have configured the bot to start the conversation.
      // showBotTyping();
      // $("#userInput").prop('disabled', true);
   
      // if you want the bot to start the conversation
      // customActionTrigger();
    });
   
  // Restore persisted state (default = open once after login)
  try {
    const saved = sessionStorage.getItem(WIDGET_STATE_KEY);
    const shouldOpen = saved === null ? true : saved === 'true';
    if ($('.widget').length) {
      if (shouldOpen) openWidget(); else closeWidget();
    }
  } catch (err) {
    if (DEBUG) console.warn('[Widget] sessionStorage unavailable', err);
    // Fallback: open on first load
    if ($('.widget').length) openWidget();
  }

  // Bind handlers defensively (avoid double-binding on re-inits)
  $(document)
    .off('click.widget', '#profile_div')
    .on('click.widget', '#profile_div', toggleWidget);
   
    // clear function to clear the chat contents of the widget.
    $("#clear").click(() => {
      $(".chats").fadeOut("normal", () => {
        $(".chats").html("");
        try { sessionStorage.removeItem('autoGreetingShown'); } catch (e) {}
        try { sessionStorage.removeItem('chat_html'); } catch (e) {}
        if (typeof applyInputDirectionFromValue === 'function') { applyInputDirectionFromValue(); }
        $(".chats").fadeIn();
        if (typeof autoGreetIfNeeded === 'function') { autoGreetIfNeeded(); }
      });
    });
   
  // close function to close the widget.
  $(document)
    .off('click.widget', '#close')
    .on('click.widget', '#close', closeWidget);

  // Ensure launcher remains visible
  setInterval(() => {
    if ($('#profile_div').length && $('#profile_div').is(':hidden')) {
      $('#profile_div').show();
    }
  }, 1500);
});
