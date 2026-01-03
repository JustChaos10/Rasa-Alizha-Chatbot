/**
 * renders the dropdown message
 * @param {Array} dropDownData dropdown data
 */
function renderDropDwon(dropDownData) {
    const dropDownDiv = `<div class="dropDownMsg">
      <select class="browser-default" id="dropDownSelect">
        <option value="" disabled selected>Choose an option</option>
      </select>
    </div>`;
   
    $(dropDownDiv).appendTo(".chats");
   
    // Add options to dropdown
    dropDownData.forEach((option) => {
      const optionElement = `<option value="${option.value}">${option.label}</option>`;
      $("#dropDownSelect").append(optionElement);
    });
   
    scrollToBottomOfResults();
  }
   
  // Handle dropdown selection
  $(document).on("change", "#dropDownSelect", function() {
    const selectedValue = $(this).val();
    const selectedText = $(this).find("option:selected").text();
   
    if (selectedValue) {
      // Remove dropdown after selection
      $(".dropDownMsg").remove();
     
      // Show user message
      setUserResponse(selectedText);
     
      // Send selected value to RASA
      send(selectedValue);
    }
  });