import { Router } from "express";
import { ChatController } from "./chat.controller";
import { asyncHandler } from "../../utils/asyncHandler";
import { authMiddleware } from "../../middlewares/auth.middleware";

const router = Router();

router.use(authMiddleware);
router.get("/", asyncHandler(ChatController.listConversations));
router.get(
  "/:conversationId/messages",
  asyncHandler(ChatController.getMessages),
);
router.post("/", asyncHandler(ChatController.chat));
router.get("/stream", asyncHandler(ChatController.streamChat));

export default router;
