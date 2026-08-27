"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ChatController = void 0;
const chat_service_1 = require("./chat.service");
const sanitize = (text) => (text ? String(text).trim() : "");
class ChatController {
    static chat(req, res) {
        return __awaiter(this, void 0, void 0, function* () {
            var _a;
            const userId = req.userId;
            if (!userId) {
                return res.status(401).json({ error: "Unauthorized" });
            }
            const prompt = sanitize((_a = req.body) === null || _a === void 0 ? void 0 : _a.prompt);
            if (!prompt) {
                return res.status(400).json({ error: "prompt is required" });
            }
            const conversation = yield chat_service_1.ChatService.getOrCreateConversation(userId);
            const userMessage = yield chat_service_1.ChatService.createMessage(conversation.id, "user", prompt);
            // TODO: replace this with actual RAG/LLM response generation.
            const assistantOutput = `Echo: ${prompt}`;
            const assistantMessage = yield chat_service_1.ChatService.createMessage(conversation.id, "assistant", assistantOutput);
            yield chat_service_1.ChatService.updateConversationTimestamp(conversation.id);
            return res.json({
                conversationId: conversation.id,
                history: [userMessage, assistantMessage],
                answer: assistantOutput,
            });
        });
    }
    static listConversations(req, res) {
        return __awaiter(this, void 0, void 0, function* () {
            const userId = req.userId;
            if (!userId) {
                return res.status(401).json({ error: "Unauthorized" });
            }
            const conversations = yield chat_service_1.ChatService.listConversations(userId);
            res.json(conversations);
        });
    }
    static getMessages(req, res) {
        return __awaiter(this, void 0, void 0, function* () {
            const userId = req.userId;
            if (!userId) {
                return res.status(401).json({ error: "Unauthorized" });
            }
            const conversationId = Array.isArray(req.params.conversationId)
                ? req.params.conversationId[0]
                : req.params.conversationId;
            const conversation = yield chat_service_1.ChatService.getConversationById(userId, conversationId);
            if (!conversation) {
                return res.status(404).json({ error: "Conversation not found" });
            }
            const messages = yield chat_service_1.ChatService.listMessages(conversationId);
            res.json(messages);
        });
    }
    static streamChat(req, res) {
        return __awaiter(this, void 0, void 0, function* () {
            const userId = req.userId;
            if (!userId) {
                return res.status(401).json({ error: "Unauthorized" });
            }
            const rawPrompt = Array.isArray(req.query.prompt) ? req.query.prompt[0] : req.query.prompt;
            const prompt = sanitize(rawPrompt);
            if (!prompt) {
                return res.status(400).json({ error: "prompt is required" });
            }
            res.writeHead(200, {
                Connection: "keep-alive",
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            });
            const conversation = yield chat_service_1.ChatService.getOrCreateConversation(userId);
            yield chat_service_1.ChatService.createMessage(conversation.id, "user", prompt);
            const mockTokens = prompt
                .split(" ")
                .map((token, index) => `Token(${index + 1}): ${token}`);
            let responseText = "";
            for (const token of mockTokens) {
                responseText += token + " ";
                res.write(`event: token\ndata: ${JSON.stringify(token)}\n\n`);
                yield new Promise((resolve) => setTimeout(resolve, 180));
            }
            const finalMessage = responseText.trim();
            yield chat_service_1.ChatService.createMessage(conversation.id, "assistant", finalMessage);
            yield chat_service_1.ChatService.updateConversationTimestamp(conversation.id);
            res.write(`event: done\ndata: ${JSON.stringify(finalMessage)}\n\n`);
            res.write(`event: close\ndata: done\n\n`);
            res.end();
        });
    }
}
exports.ChatController = ChatController;
