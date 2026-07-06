import re

with open('public/index.html', 'r') as f:
    code = f.read()

old_loop = """                    try {
                        const pStart = performance.now();
                        let currentPos = 0;
                        for (let i = 0; i < tokenIds.length - 1; i++) {
                            await mathEngine.forward(Number(tokenIds[i]), currentPos);
                            currentPos++;
                        }
                        const pEnd = performance.now();
                        // Removed the prompt processed timing bubble from chat
                        console.log(`Prompt processed in ${Math.round(pEnd - pStart)}ms`);

                        let currentToken = Number(tokenIds[tokenIds.length - 1]);
                        
                        const streamBubble = appendMessage('assistant', '');
                        streamBubble.classList.add('typing-cursor');
                        let fullText = "";
                        
                        let totalTokens = 0;
                        const tStart = performance.now();

                        for (let step = 0; step < maxTokens; step++) {
                            const bestToken = await mathEngine.forward(currentToken, currentPos);
                            currentToken = bestToken;
                            currentPos++;
                            
                            if (bestToken === 151643 || bestToken === 151645) { // EOS tokens for Qwen
                                break;
                            }
                            
                            totalTokens++;
                            const decoded = await tokenizer.decode([bestToken]);
                            fullText += decoded;
                            streamBubble.textContent = fullText;
                            
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }"""

new_loop = """                    try {
                        const pStart = performance.now();
                        let currentPos = 0;
                        const promptTokens = Array.from(tokenIds).map(Number);
                        
                        // We must chunk the prompt if it exceeds maxPrefillLen!
                        let currentToken;
                        const maxChunk = 512;
                        for (let i = 0; i < promptTokens.length; i += maxChunk) {
                            const chunk = promptTokens.slice(i, i + maxChunk);
                            currentToken = await mathEngine.forwardSequence(chunk, currentPos);
                            currentPos += chunk.length;
                        }

                        const pEnd = performance.now();
                        console.log(`Prompt processed in ${Math.round(pEnd - pStart)}ms`);
                        
                        const streamBubble = appendMessage('assistant', '');
                        streamBubble.classList.add('typing-cursor');
                        let fullText = "";
                        
                        let totalTokens = 0;
                        const tStart = performance.now();

                        for (let step = 0; step < maxTokens; step++) {
                            if (currentToken === 151643 || currentToken === 151645) { // EOS tokens for Qwen
                                break;
                            }
                            
                            totalTokens++;
                            const decoded = await tokenizer.decode([currentToken]);
                            fullText += decoded;
                            streamBubble.textContent = fullText;
                            
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                            
                            currentToken = await mathEngine.forwardSequence([currentToken], currentPos);
                            currentPos++;
                        }"""

code = code.replace(old_loop, new_loop)

with open('public/index.html', 'w') as f:
    f.write(code)

print("Patched index.html")
