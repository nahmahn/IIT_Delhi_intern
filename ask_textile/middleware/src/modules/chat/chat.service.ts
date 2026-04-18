import prisma from "../../config/prisma";
import { AppError } from "../../utils/AppError";
const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export class ChatService {
 static async getOrCreateConversation(userId: string, conversationId?: string) {
  if (conversationId && UUID_REGEX.test(conversationId)) {  // 👈 validate UUID first
    const existing = await prisma.conversation.findFirst({
      where: { id: conversationId, userId },
    });
    if (existing) return existing;
  }
  // "new-xxx" ids, missing, or not-found all fall through to create
  return prisma.conversation.create({
    data: { userId, title: "New Conversation" },
  });
}

  static async getConversationById(userId: string, conversationId: string) {
    const conversation = await prisma.conversation.findUnique({
      where: { id: conversationId },
    });

    if (!conversation || conversation.userId !== userId) {
      throw new AppError("Conversation not found", 404);
    }

    return conversation;
  }

  static async createMessage(
    conversationId: string,
    role: "user" | "assistant" | "system",
    content: string,
    tokenCount?: number,
  ) {
    return prisma.message.create({
      data: {
        conversationId,
        role,
        content,
        tokenCount,
      },
    });
  }

  static async updateConversationTimestamp(conversationId: string) {
    return prisma.conversation.update({
      where: { id: conversationId },
      data: { updatedAt: new Date() },
    });
  }

  static async listConversations(userId: string) {
    return prisma.conversation.findMany({
      where: { userId },
      orderBy: { updatedAt: "desc" },
      include: {
        messages: {
          take: 1,
          orderBy: { createdAt: "desc" },
        },
      },
    });
  }

  static async listMessages(conversationId: string) {
    return prisma.message.findMany({
      where: { conversationId },
      orderBy: { createdAt: "asc" },
    });
  }
}
