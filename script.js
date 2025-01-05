// Send message function
function sendMessage() {
    const inputMessage = document.getElementById("inputMessage").value.trim();
    const chatBox = document.getElementById("chatBox");

    if (inputMessage === "") {
        return; // Do not send empty messages
    }

    // Add the user's question to the chat
    const userMessage = document.createElement("div");
    userMessage.classList.add("message", "question");
    userMessage.textContent = inputMessage;
    chatBox.appendChild(userMessage);

    // Clear the input field
    document.getElementById("inputMessage").value = "";

    // Simulate an answer after a short delay
    setTimeout(() => {
        const botMessage = document.createElement("div");
        botMessage.classList.add("message", "answer");
        botMessage.textContent = "This is a simulated response for: " + inputMessage;
        chatBox.appendChild(botMessage);

        // Scroll to the bottom of the chat
        chatBox.scrollTop = chatBox.scrollHeight;
    }, 500);
}

// Detect "Enter" key to submit message
document.getElementById("inputMessage").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});
        