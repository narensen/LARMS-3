// Send message function
async function sendMessage() {
    const inputMessage = document.getElementById("inputMessage").value.trim();
    const chatBox = document.getElementById("chatBox");

    if (inputMessage === "") {
        return;
    }

    // Add the user's question to the chat
    const userMessage = document.createElement("div");
    userMessage.classList.add("message", "question");
    userMessage.textContent = inputMessage;
    chatBox.appendChild(userMessage);

    // Clear the input field and disable it
    document.getElementById("inputMessage").value = "";
    document.getElementById("inputMessage").disabled = true;

    try {
        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                user_question: inputMessage,
                selected_model: "llama-3.3-70b-specdec",
                temperature: 0.7
            }),
        });

        if (!response.ok) {
            // Get the error details from the response
            const errorData = await response.json();
            throw new Error(`Server error: ${errorData.detail || response.statusText}`);
        }

        const data = await response.json();
        
        if (data.ai_response) {
            const botMessage = document.createElement("div");
            botMessage.classList.add("message", "answer");
            botMessage.textContent = data.ai_response;
            chatBox.appendChild(botMessage);
        } else {
            throw new Error("No response received from AI");
        }
    } catch (error) {
        console.error("Error details:", error);
        const botMessage = document.createElement("div");
        botMessage.classList.add("message", "answer", "error");
        botMessage.textContent = `Error: ${error.message}`;
        chatBox.appendChild(botMessage);
    } finally {
        document.getElementById("inputMessage").disabled = false;
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

// Detect "Enter" key to submit message
document.getElementById("inputMessage").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});