/**
 * creates the collapsible
 * @param {Array} data collapsible data
 */
function createCollapsible(data) {
    let collapsibleData = `<ul class="collapsible">`;
   
    data.forEach((item, index) => {
      collapsibleData += `
        <li>
          <div class="collapsible-header">${item.title}</div>
          <div class="collapsible-body"><span>${item.description}</span></div>
        </li>
      `;
    });
   
    collapsibleData += `</ul>`;
   
    $(collapsibleData).appendTo(".chats");
   
    // Initialize collapsible functionality
    setTimeout(() => {
      // Simple collapsible toggle functionality
      $(".collapsible-header").off("click").on("click", function() {
        const body = $(this).next(".collapsible-body");
        const isActive = $(this).hasClass("active");
       
        // Close all other collapsibles
        $(".collapsible-header").removeClass("active");
        $(".collapsible-body").slideUp(300);
       
        if (!isActive) {
          $(this).addClass("active");
          body.slideDown(300);
        }
      });
     
      scrollToBottomOfResults();
    }, 100);
  }