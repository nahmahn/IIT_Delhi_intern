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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ChatService = void 0;
const prisma_1 = __importDefault(require("../../config/prisma"));
const AppError_1 = require("../../utils/AppError");
class ChatService {
    static getOrCreateConversation(userId) {
        return __awaiter(this, void 0, void 0, function* () {
            const existing = yield prisma_1.default.conversation.findFirst({
                where: { userId },
                orderBy: { updatedAt: "desc" },
            });
            if (existing) {
                return existing;
            }
            return yield prisma_1.default.conversation.create({
                data: {
                    userId,
                    title: "New Conversation",
                },
            });
        });
    }
    static getConversationById(userId, conversationId) {
        return __awaiter(this, void 0, void 0, function* () {
            const conversation = yield prisma_1.default.conversation.findUnique({
                where: { id: conversationId },
            });
            if (!conversation || conversation.userId !== userId) {
                throw new AppError_1.AppError("Conversation not found", 404);
            }
            return conversation;
        });
    }
    static createMessage(conversationId, role, content, tokenCount) {
        return __awaiter(this, void 0, void 0, function* () {
            return prisma_1.default.message.create({
                data: {
                    conversationId,
                    role,
                    content,
                    tokenCount,
                },
            });
        });
    }
    static updateConversationTimestamp(conversationId) {
        return __awaiter(this, void 0, void 0, function* () {
            return prisma_1.default.conversation.update({
                where: { id: conversationId },
                data: { updatedAt: new Date() },
            });
        });
    }
    static listConversations(userId) {
        return __awaiter(this, void 0, void 0, function* () {
            return prisma_1.default.conversation.findMany({
                where: { userId },
                orderBy: { updatedAt: "desc" },
                include: {
                    messages: {
                        take: 1,
                        orderBy: { createdAt: "desc" },
                    },
                },
            });
        });
    }
    static listMessages(conversationId) {
        return __awaiter(this, void 0, void 0, function* () {
            return prisma_1.default.message.findMany({
                where: { conversationId },
                orderBy: { createdAt: "asc" },
            });
        });
    }
}
exports.ChatService = ChatService;
