const action_name = "action_hello_world";
// Route all chat via Flask proxy so every bot response is JSON-logged in chatbot.log
const rasa_server_url = "/chat";
// Persist sender_id across reloads to keep Rasa tracker consistent per session
let sender_id = (typeof sessionStorage !== 'undefined' && sessionStorage.getItem('sender_id')) || uuidv4();
if (typeof sessionStorage !== 'undefined') {
  try { sessionStorage.setItem('sender_id', sender_id); } catch (e) {}
}
