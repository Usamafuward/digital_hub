document.documentElement.setAttribute("class", "dark");

document.addEventListener("DOMContentLoaded", function () {
  document.getElementById("send-button").addEventListener("click", sendMessage);
  document
    .getElementById("message-input")
    .addEventListener("keypress", function (e) {
      if (e.key === "Enter") {
        sendMessage();
      }
    });

  document
    .getElementById("voice-button")
    .addEventListener("click", toggleVoiceMode);
});

function scrollToChat() {
  setTimeout(() => {
    addQA(
      "Welcome to Support",
      "Hello! I'm your company support assistant. How can I help you today?"
    );
  }, 500);
  document
    .getElementById("chat-section")
    .scrollIntoView({ behavior: "smooth" });
  setTimeout(() => {
    document.getElementById("message-input").focus();
  }, 800);
}

function addQA(question, answer, references = null) {
  const messagesDiv = document.getElementById("chat-messages");
  const messageDiv = document.createElement("div");
  messageDiv.className = "flex justify-start w-full";

  const contentWrapper = document.createElement("div");
  contentWrapper.className = "flex flex-col w-full";

  const messageContainer = document.createElement("div");
  messageContainer.className =
    "border-l-4 border-blue-900 rounded-r-2xl p-4 w-full shadow-lg backdrop-blur-sm transition-all hover:border-blue-400";

  // Add question as topic header with improved styling
  const topicHeader = document.createElement("div");
  topicHeader.className =
    "font-bold text-blue-200 text-lg mb-3 border-b border-blue-500/30 pb-2 flex items-center";

  // Add user icon before question
  const userIcon = document.createElement("span");
  userIcon.className = "mr-2 flex-shrink-0";
  userIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
      <circle cx="12" cy="7" r="4"></circle>
    </svg>
  `;

  const questionText = document.createElement("span");
  questionText.textContent = question;

  topicHeader.appendChild(userIcon);
  topicHeader.appendChild(questionText);

  // Add answer with improved styling and assistant icon
  const messageContentWrapper = document.createElement("div");
  messageContentWrapper.className = "flex items-start";

  const assistantIcon = document.createElement("div");
  assistantIcon.className = "mr-3 mt-2.5 flex-shrink-0 text-blue-300";
  assistantIcon.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="10"></circle>
      <path d="M12 16v-4"></path>
      <path d="M12 8h.01"></path>
    </svg>
  `;

  const messageContent = document.createElement("p");
  messageContent.className = "answer-content text-white leading-relaxed";
  messageContent.textContent = answer;

  messageContentWrapper.appendChild(assistantIcon);
  messageContentWrapper.appendChild(messageContent);

  // Append elements to container
  messageContainer.appendChild(topicHeader);
  messageContainer.appendChild(messageContentWrapper);

  if (references && references.length > 0) {
    const docLinksContainer = document.createElement("div");
    docLinksContainer.className =
      "document-resources mt-4 pt-3 border-t border-blue-500/30";

    const referencesHeader = document.createElement("div");
    referencesHeader.className = "text-sm text-blue-300 mb-2 flex items-center";
    referencesHeader.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
      </svg>
      References
    `;

    const buttonContainer = document.createElement("div");
    buttonContainer.className = "flex flex-wrap gap-2";

    references.forEach((doc) => {
      const docButton = document.createElement("button");
      docButton.className =
        "text-xs bg-blue-800/50 hover:bg-blue-700 text-white px-3 py-1.5 rounded-full transition-colors flex items-center border border-blue-600";
      docButton.onclick = function () {
        showDocument(doc);
      };

      docButton.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" class="w-3 h-3 mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
        ${doc.filename}
      `;

      buttonContainer.appendChild(docButton);
    });

    docLinksContainer.appendChild(referencesHeader);
    docLinksContainer.appendChild(buttonContainer);
    messageContainer.appendChild(docLinksContainer);
  }

  // Add timestamp
  const timestamp = document.createElement("div");
  timestamp.className = "text-xs text-blue-400/60 mt-3 text-right";
  const now = new Date();
  timestamp.textContent = now.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  messageContainer.appendChild(timestamp);

  contentWrapper.appendChild(messageContainer);
  messageDiv.appendChild(contentWrapper);
  messagesDiv.appendChild(messageDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

  const typingIndicator = document.getElementById("typing-indicator");
  if (typingIndicator) {
    typingIndicator.remove();
  }

  // Add subtle fade-in animation
  messageDiv.style.opacity = "0";
  messageDiv.style.transition = "opacity 0.3s ease-in-out";

  setTimeout(() => {
    messageDiv.style.opacity = "1";
  }, 50);

  setTimeout(() => {
    messageDiv.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 500);
}

function showTypingIndicator() {
  const messagesDiv = document.getElementById("chat-messages");
  const typingDiv = document.createElement("div");
  typingDiv.id = "typing-indicator";
  typingDiv.className = "flex justify-start";
  typingDiv.innerHTML = `
    <div class="bg-blue-900/50 text-white rounded-tl-2xl rounded-tr-2xl rounded-br-2xl p-3 max-w-xs md:max-w-md shadow-md backdrop-blur-sm">
        <div class="flex space-x-2">
            <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0s"></div>
            <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
            <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.4s"></div>
        </div>
    </div>
  `;
  messagesDiv.appendChild(typingDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById("message-input");
  const text = input.value.trim();

  if (!text) return;

  const question = text;
  input.value = "";

  showTypingIndicator();

  console.log(window.BACKEND_URL);

  try {
    const response = await fetch(`${window.BACKEND_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: text }),
    });

    if (!response.ok) {
      throw new Error("Failed to get response from server");
    }

    const data = await response.json();
    addQA(question, data.answer, data.references);
  } catch (error) {
    console.error("Error:", error);
    addQA(
      question,
      "I'm sorry, I couldn't process your request. Please try again later."
    );
  }
}

let isWebRTCActive = false;
let peerConnection;
let dataChannel;
let isToggleInProgress = false;

function handleTrack(event) {
  const messagesDiv = document.getElementById("chat-messages");
  const audioDiv = document.createElement("div");
  audioDiv.className = "flex justify-start w-full mb-4";

  const audioContainer = document.createElement("div");
  audioContainer.className =
    "border-l-4 border-blue-900 rounded-r-2xl p-4 w-full md:max-w-4xl shadow-lg backdrop-blur-sm transition-all hover:border-blue-400";

  const audioHeader = document.createElement("div");
  audioHeader.className =
    "font-bold text-blue-200 text-lg mb-3 border-b border-blue-500/30 pb-2 flex items-center";
  audioHeader.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
      <line x1="12" y1="19" x2="12" y2="23"></line>
      <line x1="8" y1="23" x2="16" y2="23"></line>
    </svg>
    <span>Voice Response</span>
  `;

  const audioElement = document.createElement("audio");
  audioElement.srcObject = event.streams[0];
  audioElement.autoplay = true;
  audioElement.controls = true;
  audioElement.className = "w-full mt-2";

  audioContainer.appendChild(audioHeader);
  audioContainer.appendChild(audioElement);
  audioDiv.appendChild(audioContainer);
  messagesDiv.appendChild(audioDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function createDataChannel() {
  dataChannel = peerConnection.createDataChannel("response");

  dataChannel.addEventListener("open", () => {
    console.log("Data channel opened");
    configureData();
  });

  let pendingUserQuestion = null;
  let pendingAssistantAnswer = null;

  dataChannel.addEventListener("message", async (ev) => {
    const msg = JSON.parse(ev.data);

    if (msg.type === "conversation.item.input_audio_transcription.completed") {
      console.log("User transcript received:", msg);
      pendingUserQuestion = msg?.transcript || "";
    }

    if (msg.type === "response.output_item.done") {
      console.log("Assistant transcript received:", msg);
      pendingAssistantAnswer = msg?.item?.content?.[0]?.transcript || "";

      if (pendingUserQuestion && pendingAssistantAnswer) {
        addQA(pendingUserQuestion, pendingAssistantAnswer);
        pendingUserQuestion = null;
        pendingAssistantAnswer = null;
      }
    }
  });
}

function configureData() {
  const event = {
    type: "session.update",
    session: {
      modalities: ["text", "audio"],
    },
  };
  dataChannel.send(JSON.stringify(event));
}

function toggleVoiceMode() {
  const voiceButton = document.getElementById("voice-button");

  // Prevent multiple rapid clicks
  if (isToggleInProgress) {
    console.log("Toggle operation in progress, please wait");
    return;
  }

  isToggleInProgress = true;

  if (isWebRTCActive) {
    // First update UI to show we're processing
    voiceButton.classList.remove("recording", "bg-red-600", "hover:bg-red-700");
    voiceButton.classList.add("bg-yellow-500", "hover:bg-yellow-600"); // Intermediate state
    voiceButton.querySelector(".lucide-mic").classList.add("text-yellow-200");

    console.log("Stopping WebRTC connection...");

    // Create a promise to stop WebRTC
    const stopPromise = new Promise((resolve) => {
      // Stop all tracks first
      if (peerConnection) {
        try {
          peerConnection.getSenders().forEach((sender) => {
            if (sender.track) sender.track.stop();
          });

          peerConnection.getReceivers().forEach((receiver) => {
            if (receiver.track) receiver.track.stop();
          });
        } catch (e) {
          console.error("Error stopping tracks:", e);
        }
      }

      // Close data channel
      if (dataChannel) {
        try {
          dataChannel.close();
        } catch (e) {
          console.error("Error closing data channel:", e);
        }
        dataChannel = null;
      }

      // Close peer connection
      if (peerConnection) {
        try {
          peerConnection.close();
        } catch (e) {
          console.error("Error closing peer connection:", e);
        }
        peerConnection = null;
      }

      // Force reset state
      isWebRTCActive = false;

      // Resolve after a small timeout to ensure proper cleanup
      setTimeout(resolve, 300);
    });

    // Handle the completion of the stop operation
    stopPromise.then(() => {
      console.log("WebRTC successfully stopped");

      // Update UI back to normal state
      voiceButton.classList.remove("bg-yellow-500", "hover:bg-yellow-600");
      voiceButton
        .querySelector(".lucide-mic")
        .classList.remove("text-yellow-200", "text-red-200", "animate-pulse");
      voiceButton.classList.add("bg-blue-600", "hover:bg-blue-700");

      addQA("Voice Chat Status", "Voice chat has been deactivated.");
      isToggleInProgress = false;
    });
  } else {
    // Update UI to recording state
    voiceButton.classList.add("recording", "bg-red-600", "hover:bg-red-700");
    voiceButton.classList.remove(
      "bg-blue-600",
      "hover:bg-blue-700",
      "bg-yellow-500",
      "hover:bg-yellow-600"
    );
    voiceButton
      .querySelector(".lucide-mic")
      .classList.add("text-red-200", "animate-pulse");
    voiceButton
      .querySelector(".lucide-mic")
      .classList.remove("text-yellow-200");

    console.log("Starting WebRTC connection...");

    // Start WebRTC
    startWebRTC()
      .then(() => {
        console.log("WebRTC connection established");
        isToggleInProgress = false;
      })
      .catch((error) => {
        console.error("Failed to start WebRTC:", error);

        // Reset UI on error
        voiceButton.classList.remove(
          "recording",
          "bg-red-600",
          "hover:bg-red-700",
          "bg-yellow-500",
          "hover:bg-yellow-600"
        );
        voiceButton.classList.add("bg-blue-600", "hover:bg-blue-700");
        voiceButton
          .querySelector(".lucide-mic")
          .classList.remove("text-red-200", "animate-pulse", "text-yellow-200");

        addQA(
          "Voice Chat Error",
          "Failed to start voice chat. Please check your microphone permissions."
        );
        isToggleInProgress = false;
      });
  }
}

async function startWebRTC() {
  if (isWebRTCActive || peerConnection || dataChannel) {
    console.log("WebRTC resources exist, forcing cleanup first");
    await new Promise((resolve) => {
      // Properly clean up existing resources
      try {
        if (peerConnection) {
          peerConnection.getSenders().forEach((sender) => {
            if (sender.track) sender.track.stop();
          });

          peerConnection.getReceivers().forEach((receiver) => {
            if (receiver.track) receiver.track.stop();
          });

          peerConnection.close();
          peerConnection = null;
        }

        if (dataChannel) {
          dataChannel.close();
          dataChannel = null;
        }

        isWebRTCActive = false;
      } catch (e) {
        console.error("Error in forced cleanup:", e);
      }

      // Small delay to ensure cleanup is complete
      setTimeout(resolve, 300);
    });
  }

  try {
    peerConnection = new RTCPeerConnection();
    peerConnection.ontrack = handleTrack;
    createDataChannel();

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream
      .getTracks()
      .forEach((track) =>
        peerConnection.addTransceiver(track, { direction: "sendrecv" })
      );

    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);

    const response = await fetch(`${window.BACKEND_URL}/rtc-connect`, {
      method: "POST",
      body: offer.sdp,
      headers: { "Content-Type": "application/sdp" },
    });

    if (!response.ok) {
      throw new Error("Failed to connect to server");
    }

    const answer = await response.text();
    await peerConnection.setRemoteDescription({
      sdp: answer,
      type: "answer",
    });

    isWebRTCActive = true;
    addQA(
      "Voice Chat Status",
      "Voice chat activated. You can start speaking now."
    );
    return true;
  } catch (error) {
    console.error("WebRTC Error:", error);

    // Ensure everything is cleaned up on error
    if (peerConnection) {
      try {
        peerConnection.close();
      } catch (e) {
        console.error("Error closing peer connection:", e);
      }
      peerConnection = null;
    }

    if (dataChannel) {
      try {
        dataChannel.close();
      } catch (e) {
        console.error("Error closing data channel:", e);
      }
      dataChannel = null;
    }

    isWebRTCActive = false;
    throw error;
  }
}

function showDocument(doc) {
  const docViewer = document.getElementById("document-viewer");
  const chatContainer = document.getElementById("chat-container");

  docViewer.classList.remove("hidden");
  docViewer.classList.add("document-viewer-visible");

  if (window.innerWidth >= 768) {
    chatContainer.classList.add("chat-container-minimized");
  }

  if (window.innerWidth < 768) {
    docViewer.classList.add("fixed", "inset-0", "z-50", "p-4", "bg-black/90");
  }

  let contentHtml = "";
  const fileType = doc.file_type || "unknown";

  if (doc.file && doc.file.length > 0) {
    if (fileType === "pdf") {
      contentHtml = `
        <div class="h-full flex flex-col">
          <iframe 
            class="w-full h-full rounded-lg border border-blue-500/30" 
            src="data:application/pdf;base64,${doc.file}" 
            type="application/pdf"
          ></iframe>
        </div>
      `;
    } else if (fileType === "docx" || fileType === "doc") {
      contentHtml = `
        <div class="border-l-4 border-blue-500 rounded-r-2xl p-4 mt-4 shadow-lg backdrop-blur-sm">
            <div class="text-sm text-blue-300 mb-3 flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
            </svg>
            Document Preview
            </div>
            <p class="text-white leading-relaxed">${
              doc.content_preview || "Document preview not available"
            }</p>
            <div class="flex items-center mt-3 text-blue-400 text-sm">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                <line x1="16" y1="2" x2="16" y2="6"></line>
                <line x1="8" y1="2" x2="8" y2="6"></line>
                <line x1="3" y1="10" x2="21" y2="10"></line>
            </svg>
            Page ${doc.page}
            </div>
            <a 
            class="inline-flex items-center mt-4 bg-blue-600/50 hover:bg-blue-500 text-white py-2 px-4 rounded border border-blue-500 transition-colors"
            href="data:application/octet-stream;base64,${doc.file}" 
            download="${doc.filename}"
            >
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            Download Document
            </a>
        </div>
        `;
    } else {
      contentHtml = `
        <div class="border-l-4 border-blue-500 rounded-r-2xl p-4 mt-4 shadow-lg backdrop-blur-sm">
          <div class="text-sm text-blue-300 mb-3 flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
            </svg>
            Document Preview
          </div>
          <p class="text-white leading-relaxed">${
            doc.content_preview || "Document preview not available"
          }</p>
          <div class="flex items-center mt-3 text-blue-400 text-sm">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
              <line x1="16" y1="2" x2="16" y2="6"></line>
              <line x1="8" y1="2" x2="8" y2="6"></line>
              <line x1="3" y1="10" x2="21" y2="10"></line>
            </svg>
            Page ${doc.page}
          </div>
          <a 
            class="inline-flex items-center mt-4 bg-blue-600/50 hover:bg-blue-500 text-white py-2 px-4 rounded border border-blue-500 transition-colors"
            href="data:application/octet-stream;base64,${doc.file}" 
            download="${doc.filename}"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            Download Document
          </a>
        </div>
      `;
    }
  } else {
    contentHtml = `
      <div class="border-l-4 border-blue-500 rounded-r-2xl p-4 mt-4 shadow-lg backdrop-blur-sm">
        <div class="text-sm text-blue-300 mb-3 flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="12"></line>
              <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          Notice
        </div>
        <p class="text-white">Document content is not available for preview.</p>
        <div class="flex items-center mt-3 text-blue-400 text-sm">
          <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
            <line x1="16" y1="2" x2="16" y2="6"></line>
            <line x1="8" y1="2" x2="8" y2="6"></line>
            <line x1="3" y1="10" x2="21" y2="10"></line>
          </svg>
          Page ${doc.page}
        </div>
      </div>
    `;
  }

  docViewer.innerHTML = `
    <div class="h-full flex flex-col">
        <div class="flex items-center justify-between mb-4 pb-3 border-b border-blue-500/30">
            <h3 class="text-xl font-bold text-white flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 mr-2 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
                ${doc.filename}
            </h3>
            <button id="close-doc-btn" class="bg-blue-800/30 hover:bg-blue-700/50 text-blue-300 hover:text-white transition-colors p-2 rounded-full border border-blue-600/50">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
        </div>
        <div class="flex-grow overflow-auto text-gray-300 space-y-4">
            ${contentHtml}
        </div>
    </div>
    `;

  // Add fade-in animation
  docViewer.style.opacity = "0";
  docViewer.style.transition = "opacity 0.3s ease-in-out";

  setTimeout(() => {
    docViewer.style.opacity = "1";
  }, 50);

  document
    .getElementById("close-doc-btn")
    .addEventListener("click", closeDocument);
}

function closeDocument() {
  const docViewer = document.getElementById("document-viewer");
  const chatContainer = document.getElementById("chat-container");

  // Add fade-out animation
  docViewer.style.opacity = "0";
  docViewer.style.transition = "opacity 0.3s ease-in-out";

  setTimeout(() => {
    docViewer.classList.add("hidden");
    docViewer.classList.remove("document-viewer-visible");
    docViewer.classList.remove(
      "fixed",
      "inset-0",
      "z-50",
      "p-4",
      "bg-black/90"
    );
    chatContainer.classList.remove("chat-container-minimized");
    docViewer.innerHTML = "";
  }, 300);
}
