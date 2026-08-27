import { Request, Response } from "express";
import { ChatService } from "./chat.service";

const sanitize = (text: unknown) => (text ? String(text).trim() : "");

export class ChatController {
  static async chat(req: Request, res: Response) {
  const userId = req.userId;
  if (!userId) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  const prompt = sanitize(req.body?.prompt);
  const history = Array.isArray(req.body?.history) ? req.body.history : [];
  const summary = sanitize(req.body?.summary);
  const conversationId = sanitize(req.body?.conversationId); // 👈 extract this

  if (!prompt) {
    return res.status(400).json({ error: "prompt is required" });
  }

  // 👇 Pass conversationId — if it's a "new-xxx" temp ID or missing, service creates a new one
  const isNewConversation = !conversationId || conversationId.startsWith("new-");
  const conversation = await ChatService.getOrCreateConversation(
    userId,
    isNewConversation ? undefined : conversationId, // 👈 undefined = force create new
  );

    const userMessage = await ChatService.createMessage(
      conversation.id,
      "user",
      prompt,
    );

    // Example RAG/LLM behavior: build prompt context from history + summary
    // TODO: replace this block with real model call.
    const contextPayload = `Summary: ${summary || "(none)"}\nHistory: ${JSON.stringify(
      history,
    )}\nCurrent: ${prompt}`;
    const assistantOutput = `Mock response based on prompt and context:\n${contextPayload}`;

    const assistantMessage = await ChatService.createMessage(
      conversation.id,
      "assistant",
      assistantOutput,
    );

    await ChatService.updateConversationTimestamp(conversation.id);

    return res.json({
      conversationId: conversation.id,
      history: [userMessage, assistantMessage],
      answer: assistantOutput,
    });
  }

  static async listConversations(req: Request, res: Response) {
    const userId = req.userId;
    if (!userId) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    const conversations = await ChatService.listConversations(userId);
    res.json(conversations);
  }

  static async getMessages(req: Request, res: Response) {
    const userId = req.userId;
    if (!userId) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    const conversationId = Array.isArray(req.params.conversationId)
      ? req.params.conversationId[0]
      : req.params.conversationId;

    const conversation = await ChatService.getConversationById(
      userId,
      conversationId,
    );
    if (!conversation) {
      return res.status(404).json({ error: "Conversation not found" });
    }

    const messages = await ChatService.listMessages(conversationId);
    res.json(messages);
  }

  static async streamChat(req: Request, res: Response) {
    const userId = req.userId;
    if (!userId) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    const rawPrompt = Array.isArray(req.query.prompt)
      ? req.query.prompt[0]
      : req.query.prompt;
    const prompt = sanitize(rawPrompt);
    if (!prompt) {
      return res.status(400).json({ error: "prompt is required" });
    }

    res.writeHead(200, {
      Connection: "keep-alive",
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
    });

    const conversation = await ChatService.getOrCreateConversation(userId);
    await ChatService.createMessage(conversation.id, "user", prompt);

    const mockTokens = prompt
      .split(" ")
      .map((token, index) => `Token(${index + 1}): ${token}`);
    let responseText = "";

    for (const token of mockTokens) {
      responseText += token + " ";
      res.write(`event: token\ndata: ${JSON.stringify(token)}\n\n`);
      await new Promise((resolve) => setTimeout(resolve, 180));
    }

    const finalMessage = responseText.trim();
    await ChatService.createMessage(conversation.id, "assistant", finalMessage);
    await ChatService.updateConversationTimestamp(conversation.id);

    res.write(`event: done\ndata: ${JSON.stringify(finalMessage)}\n\n`);
    res.write(`event: close\ndata: done\n\n`);
    res.end();
  }
}
