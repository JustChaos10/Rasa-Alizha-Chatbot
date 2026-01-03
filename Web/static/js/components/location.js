/**
 * gets the location of the user
 */
function getLocation() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(showPosition, showError);
  } else {
    const errorMsg = "Geolocation is not supported by this browser.";
    const locationResponse = `<img class="botAvatar" src="./static/images/aliza-icon.jpg"/><p class="botMsg">${errorMsg}</p><div class="clearfix"></div>`;
    $(locationResponse).appendTo(".chats").hide().fadeIn(1000);
    scrollToBottomOfResults();
  $(".usrInput").prop("disabled", false);
  }
}
 
/**
 * shows the position of the user
 * @param {Object} position user position
 */
function showPosition(position) {
  const lat = position.coords.latitude;
  const lon = position.coords.longitude;
 
  const locationMsg = `📍 Your location: Latitude ${lat.toFixed(6)}, Longitude ${lon.toFixed(6)}`;
  const locationResponse = `<img class="botAvatar" src="./static/images/aliza-icon.jpg"/><p class="botMsg">${locationMsg}</p><div class="clearfix"></div>`;
 
  $(locationResponse).appendTo(".chats").hide().fadeIn(1000);
  scrollToBottomOfResults();
 
  // Send location data to RASA
  const locationData = `Location: ${lat}, ${lon}`;
  send(locationData);
 
  $(".usrInput").prop("disabled", false);
}
 
/**
 * shows error if location access is denied
 * @param {Object} error location error
 */
function showError(error) {
  let errorMsg = "";
 
  switch(error.code) {
    case error.PERMISSION_DENIED:
      errorMsg = "Location access denied by user.";
      break;
    case error.POSITION_UNAVAILABLE:
      errorMsg = "Location information is unavailable.";
      break;
    case error.TIMEOUT:
      errorMsg = "Location request timed out.";
      break;
    default:
      errorMsg = "An unknown error occurred while retrieving location.";
      break;
  }
 
  const locationResponse = `<img class="botAvatar" src="./static/images/aliza-icon.jpg"/><p class="botMsg">❌ ${errorMsg}</p><div class="clearfix"></div>`;
  $(locationResponse).appendTo(".chats").hide().fadeIn(1000);
  scrollToBottomOfResults();
 
  $(".usrInput").prop("disabled", false);
}